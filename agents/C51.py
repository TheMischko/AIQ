import atexit
import binascii
import math
import os
import random


import numpy as np
import torch
from torch import nn

from .Agent import Agent
from .neural_utils.IDeepQLAgent import IDeepQLAgent
from .neural_utils.neuralNet import get_optimizer, get_criterion
from .neural_utils.plottingTools import PlottingTools
from .neural_utils.distributionNeuralNet import DistributionNeuralNet
from .neural_utils.replayMemory import ReplayMemory


class C51(IDeepQLAgent):
    GRID_MIN_VAL = -5.
    GRID_MAX_VAL = 5.

    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, grid_size, tau):
        Agent.__init__(self, refm, disc_rate)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = math.floor(batch_size)
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.num_atoms = math.floor(grid_size)
        self.atom_step = ((self.GRID_MAX_VAL - self.GRID_MIN_VAL) / (self.num_atoms - 1))
        self.value_range = np.linspace(self.GRID_MIN_VAL, self.GRID_MAX_VAL, self.num_atoms)
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
        self.neural_size_l1 = 128
        self.neural_size_l2 = 64
        self.neural_size_l3 = 32

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
        # Replay buffer
        self.memory = ReplayMemory(10000)
        # Policy net
        self.policy_net = DistributionNeuralNet(self.neural_input_size, self.neural_size_l1, self.neural_size_l2,
                                                self.neural_size_l3, self.num_actions, self.num_atoms)
        self.target_net = DistributionNeuralNet(self.neural_input_size, self.neural_size_l1, self.neural_size_l2,
                                                self.neural_size_l3, self.num_actions, self.num_atoms)
        self.optimizer = get_optimizer(self.target_net, learning_rate=self.learning_rate)

    def computeActionFromQValue(self, state):
        with torch.no_grad():
            q_vals = self.target_net(state)
            distributions = q_vals.view(self.num_actions, self.num_atoms)
            range_view = self.value_range.reshape(1, 1, -1)
            if self.steps_done > 150 and True:
                for i in range(self.num_actions):
                    self.plotting_tools.plot_array(distributions[i], self.value_range, title="action %i" % i)
            q_means = torch.sum(distributions * range_view, dim=2)
            best_q_value = q_means.max()
            self.q_values_arr.append(best_q_value)
            best_action = q_means.argmax()
            return best_action.item()

    def learn_from_experience(self):
        if len(self.memory) < self.batch_size:
            return
        self.optimizer.zero_grad()
        # Sample batch from Replay memory
        state_batch, action_batch, reward_batch, next_state_batch = self.get_learning_batches()

        # Evaluate state_batch onto neural net and select distributions of selected actions
        net_values = self.policy_net(state_batch)
        q_distributions = self.get_distribution_samples(net_values, action_batch)

        # a* <- argmax_a Q(x_t+1, a)
        # Get distributions for next states
        q_next = self.target_net(next_state_batch).detach()
        q_next_splitted = q_next.reshape(q_next.size(0), self.num_actions, self.num_atoms)
        range_view = self.value_range.reshape(1, 1, -1)
        q_next_mean = torch.sum(q_next_splitted * range_view, dim=2)
        next_best_actions = q_next_mean.argmax(dim=1)
        q_next_distributions = self.get_distribution_samples(q_next, next_best_actions)

        # m_i = 0, i in 0,...,N-1
        # Set up empty matrix for target values
        q_target = torch.zeros(q_distributions.size())

        # Compute projection of T^_z_j onto the support {z_i}

        # Compute support
        # T^_z_j <- [r_t + gamma*z_j]
        next_value_range = np.expand_dims(reward_batch, 1) \
                           + self.gamma * np.expand_dims(self.value_range, 0)
        next_value_range = np.clip(next_value_range, self.GRID_MIN_VAL, self.GRID_MAX_VAL)
        # Compute positions
        next_value_position = torch.zeros(next_value_range.shape)
        # b_j <- (T^_z_j - Vmin) / delta z
        next_value_position = (next_value_range - self.GRID_MIN_VAL) / self.atom_step
        # Lower bound of relative position l
        lower_bound = np.floor(next_value_position).astype(int)
        # Upper bound of relative position u
        upper_bound = np.ceil(next_value_position).astype(int)

        # Compute the target assignment
        for i in range(net_values.size(0)):
            for j in range(self.num_atoms):
                q_target[i, 0, lower_bound[i, j]] += q_next_distributions[i, 0, j] * (upper_bound[i, j] - next_value_position[i, j])
                q_target[i, 0, upper_bound[i, j]] += q_next_distributions[i, 0, j] * (next_value_position[i, j] - lower_bound[i, j])

        # Calc loss
        #log = -torch.log(q_distributions)
        #loss = q_target * log
        #loss = torch.mean(loss)

        # Calc importance weighted loss
        #b_w = torch.tensor(np.ones_like(reward_batch))
        #loss = torch.mean(b_w*loss)
        loss = self.criterion(q_distributions, q_target)
        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss
        self.last_losses.append(torch.mean(loss).item())

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.update_interval_length == 0:
            self.copy_network_weights()

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
        target_net_state_dict = self.policy_net.state_dict()
        policy_net_state_dict = self.target_net.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau \
                                         + target_net_state_dict[key] * (1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)