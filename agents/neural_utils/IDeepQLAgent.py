import math
import random
import numpy as np
import torch

import binascii
import os

from ..Agent import Agent
from agents.neural_utils.neuralNet import NeuralNet
from agents.neural_utils.neuralNet import get_optimizer
from agents.neural_utils.neuralNet import get_criterion
from agents.neural_utils.replayMemory import ReplayMemory, Transition
from agents.neural_utils.plottingTools import PlottingTools


class IDeepQLAgent(Agent):
    MIN_EPSILON = 0
    REWARD_DIVIDER = 100
    SHOW_GRAPHS = False

    def __init__(self, refm, disc_rate, learning_rate, starting_epsilon, batch_size, tau, epsilon_decay_length):
        Agent.__init__(self, refm, disc_rate)
        self.optimizer = None
        self.policy_net = None
        self.target_net = None
        self.memory = None
        self.ref_machine = refm
        self.num_states = refm.getNumObs()  # assuming that states = observations
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells = refm.getNumObsCells()
        self.state_vec_size = self.obs_cells * self.obs_symbols
        self.neural_input_size = self.state_vec_size * 2

        self.learning_rate = learning_rate
        self.starting_epsilon = starting_epsilon
        self.epsilon = starting_epsilon
        self.episodes_till_min_decay = epsilon_decay_length
        self.batch_size = math.floor(batch_size)
        self.tau = tau
        self.criterion = get_criterion()

        self.cached_state_raw = None
        self.prev_state = None
        self.prev_action = None
        self.TERMINAL_TOKEN = "TERMINAL"
        self.steps_done = 0

        # Plotting data
        self.last_losses = list()
        self.q_values_arr = list()
        self.id = binascii.b2a_hex(os.urandom(8))
        self.last_network_output = None

        self.plotting_tools = PlottingTools()
        self.epsilon_linear_decay = (starting_epsilon - self.MIN_EPSILON) / self.episodes_till_min_decay

    def reset(self):
        self.memory = ReplayMemory(10000)
        # Network evaluating Q function
        self.target_net = NeuralNet(self.neural_input_size, self.num_actions)
        # Network that is learning from replay memory
        self.optimizer = get_optimizer(self.target_net, learning_rate=self.learning_rate)
        self.steps_done = 0
        self.epsilon = self.starting_epsilon
        self.q_values_arr.clear()
        self.last_losses.clear()

    def perceive(self, observations, reward):
        new_state_tensor = self.transferObservationToStateVec(observations)
        new_state_unsqueezed = new_state_tensor.unsqueeze(0)
        # Add to replay memory
        if (self.prev_state is not None) and (self.prev_action is not None):
            self.memory.push(
                self.prev_state,
                self.prev_action,
                new_state_unsqueezed,
                torch.tensor(reward / self.REWARD_DIVIDER, dtype=torch.float32).unsqueeze(0)
            )

        # Do learning logic
        self.learn_from_experience()

        # Get action
        opt_action = self.getAction(new_state_tensor)

        # Cache current state and selected action
        self.prev_action = torch.tensor(opt_action).unsqueeze(0).unsqueeze(0)
        self.cached_state_raw = observations
        self.prev_state = new_state_unsqueezed
        self.steps_done += 1

        return opt_action

    def episode_ended(self):
        losses = np.array(self.last_losses)
        if self.SHOW_GRAPHS:
            self.plotting_tools.plot_array(np.array(self.q_values_arr), "Q values")
            self.plotting_tools.add_values_to_average_arr(losses)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, random action will be taken or
          otherwise the best policy action will be taken.
        """
        is_random = random.random() < self.epsilon
        legal_actions = [action for action in range(self.num_actions)]
        if is_random:
            return random.choice(legal_actions)
        return self.computeActionFromQValue(state)

    def computeActionFromQValue(self, state):
        with torch.no_grad():
            action_values = self.target_net.forward(state).tolist()
            best_q_value = np.max(action_values)
            self.q_values_arr.append(best_q_value)
            policy = np.argmax(action_values)
            return policy

    def transferObservationToStateVec(self, observations):
        if len(observations) != self.obs_cells:
            raise Exception("Observation is not in count as observation cells.")

        state_vec = torch.zeros(self.neural_input_size, dtype=torch.float32)
        if self.cached_state_raw is not None:
            for i in range(self.obs_cells):
                index = self.cached_state_raw[i] + i*self.obs_symbols
                state_vec[index] = 1

        for i in range(self.obs_cells):
            index = observations[i] + (i+self.obs_cells)*self.obs_symbols
            state_vec[index] = 1
        return state_vec

    def get_learning_batches(self):
        """
        Samples batches of self.batch_size size of states, actions, rewards
        and next state values.
        :return: tuple with values as (state_batch, action_batch, reward_batch, next_states)
        """
        # Get random sample from replay memory
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch
        # This converts batch-array of Transitions to Transition of batch-arrays
        batch = Transition(*zip(*transitions))
        # Connects values from transition into single arrays
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        return state_batch, action_batch, reward_batch, next_states

    def decrement_epsilon(self):
        """
        Decrements epsilon by calculated step until it hits self.MIN_EPSILON
        """
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon -= self.epsilon_linear_decay
        else:
            self.epsilon = self.MIN_EPSILON

    def learn_from_experience(self):
        """
        Function that should handle learning of agent's neural net.
        """
        raise NotImplementedError()
