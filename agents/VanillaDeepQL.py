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


class VanillaDeepQL(IDeepQLAgent):
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

    def learn_from_experience(self):
        if len(self.memory) < self.batch_size:
            return
        self.optimizer.zero_grad()
        # Sample batch from Replay memory
        state_batch, action_batch, reward_batch, next_state_batch = self.get_learning_batches()

        # Compute Q value
        # The model computes Q(s_t), then we select the columns of actions taken.
        q_values = self.target_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states with Bellman equation.
        # Selecting their best reward with max(1)[0].
        q_next_values = None
        with torch.no_grad():
            q_next_values = reward_batch + self.disc_rate * self.target_net(next_state_batch).max(1)[0]

        # Compute loss between neural net's Q value of action taken and result of bellman equation for next state.
        loss = self.criterion(q_values, q_next_values.unsqueeze(1))

        # Optimize the model
        loss.backward()
        self.optimizer.step()

        # Store loss
        self.last_losses.append(loss.item())

        # Update epsilon
        self.decrement_epsilon()
