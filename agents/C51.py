import atexit
import binascii
import math
import os
import random


import numpy as np
import torch
from torch import nn

from .Agent import Agent
from .deep_ql.neural_utils.IDeepQLAgent import IDeepQLAgent
from .deep_ql.neural_utils.neuralNet import get_optimizer, get_criterion
from agents.deep_ql.neural_utils.plottingTools import PlottingTools
from .deep_ql.neural_utils.distributionNeuralNet import DistributionNeuralNet
from .deep_ql.neural_utils.replayMemory import ReplayMemory


class C51(IDeepQLAgent):

    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, grid_size,
                 tau, min_max_variance, neural_hidden_size):
        Agent.__init__(self, refm, disc_rate)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = math.floor(batch_size)
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.epsilon = epsilon
        self.num_atoms = math.floor(grid_size)
        self.grid_min_val = -1 * float(min_max_variance)
        self.grid_max_val = float(min_max_variance)
        self.atom_step = ((self.grid_max_val - self.grid_min_val) / (self.num_atoms - 1))
        self.value_range = np.linspace(self.grid_min_val, self.grid_max_val, self.num_atoms)
        self.tau = tau

        # Properties
        self.memory = None
        self.optimizer = None
        self.prev_state = None
        self.prev_action = None
        self.cached_state_raw = None
        self.steps_done = 0
        self.update_interval_length = 20

        # Problem domain
        self.ref_machine = refm
        self.num_states = refm.getNumObs()  # assuming that states = observations
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells = refm.getNumObsCells()
        self.state_vec_size = self.obs_cells * self.obs_symbols

        # Neural net settings
        self.neural_input_size = self.state_vec_size * 2
        self.neural_output_size = self.num_actions * self.num_atoms
        self.neural_size_hidden = int(neural_hidden_size)


        # Plotting data
        self.last_losses = list()
        self.q_values_arr = list()
        self.id = binascii.b2a_hex(os.urandom(8))
        self.last_network_output = None
        self.actions_taken = list()
        self.rewards_given = list()
        self.plotting_tools = PlottingTools()
        atexit.register(self.plotting_tools.on_exit)

    def reset(self):
        if len(self.last_losses) and False:
            self.plotting_tools.plot_array(self.last_losses, title="Loss")
            self.last_losses.clear()
        # Replay buffer
        self.memory = ReplayMemory(10000)
        # Policy net
        self.policy_net = DistributionNeuralNet(self.neural_input_size, self.num_actions, self.num_atoms,
                                                self.neural_size_hidden)
        self.target_net = DistributionNeuralNet(self.neural_input_size, self.num_actions, self.num_atoms,
                                                self.neural_size_hidden)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), self.learning_rate)

    def computeActionFromQValue(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0)
            dist = self.target_net(state)
            q_values = (dist * self.value_range).sum(dim=-1)
            action = torch.argmax(q_values, dim=-1).item()
            return action

    def learn_from_experience(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from Replay memory
        state_batch, action_batch, reward_batch, next_state_batch = self.get_learning_batches()

        # a* <- argmax_a Q(x_t+1, a)
        # Get distributions for next state actions
        with torch.no_grad():
            target_dist = self.target_net(next_state_batch)
            q_values = (target_dist * self.value_range).sum(dim=-1)
            # Create array of best actions
            max_q_actions = q_values.argmax(dim=-1)
            # Select distributions of best actions for whole batch
            max_q_dist = target_dist[range(self.batch_size), max_q_actions]
            target_atoms = reward_batch.unsqueeze(1) * self.gamma * self.value_range
            target_atoms = torch.clamp(target_atoms, self.grid_min_val, self.grid_max_val)
            # b <- (T^_z - Vmin) / delta z
            bounds = (target_atoms - self.grid_min_val) / self.atom_step
            # Lower bound of relative position l
            lower_bound = torch.floor(bounds).long()
            # Upper bound of relative position u
            upper_bound = torch.ceil(bounds).long()
            # Compute the target assignment
            prob_mass = torch.zeros(self.batch_size, self.num_atoms)
            mass = torch.zeros(self.batch_size, self.num_atoms)
            for i in range(self.batch_size):
                for j in range(self.num_atoms):
                    prob_mass[i, upper_bound[i, j]] += max_q_dist[i, j] * (bounds[i, j] - lower_bound[i, j].float())
                    prob_mass[i, lower_bound[i, j]] += max_q_dist[i, j] * (upper_bound[i, j].float() - bounds[i, j])
            prob_mass /= torch.sum(prob_mass, dim=1, keepdim=True)

        self.optimizer.zero_grad()
        # Evaluate state_batch onto trained neural net
        net_dist = self.policy_net(state_batch)
        log_probs = torch.log(net_dist[range(self.batch_size), action_batch.squeeze()])
        loss = self.criterion(log_probs, prob_mass)
        loss.backward()
        self.optimizer.step()

        # Store loss
        self.last_losses.append(torch.mean(loss).item())

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.update_interval_length == 0:
            self.copy_network_weights()

        # q_next = self.target_net(next_state_batch).detach()
        # q_next_splitted = q_next.reshape(q_next.size(0), self.num_actions, self.num_atoms)
        # range_view = self.value_range.reshape(1, 1, -1)
        # q_next_mean = torch.sum(q_next_splitted * range_view, dim=2)
        # next_best_actions = q_next_mean.argmax(dim=1)
        # q_next_distributions = self.get_distribution_samples(q_next, next_best_actions)
        #
        # # m_i = 0, i in 0,...,N-1
        # # Set up empty matrix for target values
        # q_target = torch.zeros(q_distributions.size())
        #
        # # Compute projection of T^_z_j onto the support {z_i}
        #
        # # Compute support
        # # T^_z_j <- [r_t + gamma*z_j]
        # next_value_range = np.expand_dims(reward_batch, 1) \
        #                    + self.gamma * np.expand_dims(self.value_range, 0)
        # next_value_range = np.clip(next_value_range, self.GRID_MIN_VAL, self.GRID_MAX_VAL)
        # # Compute positions
        # next_value_position = torch.zeros(next_value_range.shape)
        # # b_j <- (T^_z_j - Vmin) / delta z
        # next_value_position = (next_value_range - self.GRID_MIN_VAL) / self.atom_step
        # # Lower bound of relative position l
        # lower_bound = np.floor(next_value_position).astype(int)
        # # Upper bound of relative position u
        # upper_bound = np.ceil(next_value_position).astype(int)
        #
        # # Compute the target assignment
        # for i in range(net_values.size(0)):
        #     for j in range(self.num_atoms):
        #         q_target[i, 0, lower_bound[i, j]] += q_next_distributions[i, 0, j] * (upper_bound[i, j] - next_value_position[i, j])
        #         q_target[i, 0, upper_bound[i, j]] += q_next_distributions[i, 0, j] * (next_value_position[i, j] - lower_bound[i, j])
        #
        # # Calc loss
        # #log = -torch.log(q_distributions)
        # #loss = q_target * log
        # #loss = torch.mean(loss)
        #
        # # Calc importance weighted loss
        # #b_w = torch.tensor(np.ones_like(reward_batch))
        # #loss = torch.mean(b_w*loss)
        # loss = self.criterion(q_distributions, q_target)
        # # backprop loss
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

    def get_distribution_samples(self, neural_net_output, action_batch):
        distributions = torch.zeros(neural_net_output.size(0), 1, self.num_atoms)
        for i in range(neural_net_output.size(0)):
            action_index = action_batch[i].item()
            start_index = action_index * self.num_atoms
            end_index = action_index * self.num_atoms + self.num_atoms
            distribution = neural_net_output[0][start_index: end_index]
            distributions[i] = distribution.unsqueeze(0)
        return distributions

    def copy_network_weights(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau \
                                         + target_net_state_dict[key] * (1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

    def __str__(self):
        return "C51"
