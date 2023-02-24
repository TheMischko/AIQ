#
# Trivial agent that takes random actions
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3
#
import math
import random
import numpy as np
import torch

import binascii
import os

from .Agent import Agent
from agents.neural_utils.IDeepQLAgent import IDeepQLAgent
from agents.neural_utils.neuralNet import NeuralNet
from agents.neural_utils.neuralNet import get_optimizer
from agents.neural_utils.neuralNet import get_criterion
from agents.neural_utils.replayMemory import ReplayMemory, Transition
from agents.neural_utils.plottingTools import PlottingTools


class DeepQLNoDecay(IDeepQLAgent):
    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, neural_size_l1,
                 neural_size_l2, neural_size_l3, tau, update_interval_length):
        IDeepQLAgent.__init__(self, refm, disc_rate, learning_rate, gamma, batch_size, 0,
                              neural_size_l1, neural_size_l2, neural_size_l3)
        self.epsilon = epsilon
        self.starting_epsilon = epsilon
        self.update_interval_length = update_interval_length
        self.tau = tau

    def reset(self):
        IDeepQLAgent.reset(self)
        # Second network that is learning from replay memory
        # and target net params are copied after number of iterations
        self.policy_net = NeuralNet(self.neural_input_size, self.num_actions, self.neural_size_l1, self.neural_size_l2, self.neural_size_l3)
        self.optimizer = get_optimizer(self.policy_net, learning_rate=self.learning_rate)

    def learn_from_experience(self):
        if len(self.memory) < self.batch_size:
            return
        self.optimizer.zero_grad()
        # Sample batch from Replay memory
        state_batch, action_batch, reward_batch, next_state_batch = self.get_learning_batches()

        # Compute Q value
        # The model computes Q(s_t), then we select the columns of actions taken.
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Selecting their best reward with max(1)[0].
        q_next_values = None
        with torch.no_grad():
            q_next_values = reward_batch + self.gamma * self.target_net(next_state_batch).max(1)[0]

        # Compute loss
        loss = self.criterion(q_values, q_next_values.unsqueeze(1))

        # Optimize the model
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # Store loss
        self.last_losses.append(loss.item())

        # Update epsilon
        self.decrement_epsilon()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.update_interval_length == 0:
            self.copy_network_weights()

    def copy_network_weights(self):
        target_net_state_dict = self.policy_net.state_dict()
        policy_net_state_dict = self.target_net.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau \
                                         + target_net_state_dict[key] * (1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

    def __str__(self):
        gamma_string = str(self.gamma)
        gamma_string = gamma_string.replace(".", "_")

        epsilon_string = str(self.epsilon)
        epsilon_string = epsilon_string.replace(".", "_")
        return "DeepQL-gamma-%s-epsilon-%s" % (gamma_string, epsilon_string)