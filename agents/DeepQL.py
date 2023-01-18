#
# Trivial agent that takes random actions
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3
#
import math
import random
from random import randint

import numpy as np
import torch

from .Agent import Agent
from agents.neural_utils.neuralNet import NeuralNet
from agents.neural_utils.neuralNet import get_optimizer
from agents.neural_utils.neuralNet import get_criterion
from agents.neural_utils.replayMemory import ReplayMemory, Transition


class DeepQL(Agent):
    UPDATE_INTERVAL_LENGTH = 100
    EPSILON_LINEAR_DECAY = 0.0001
    MIN_EPSILON = 0.2
    LOG_LOSS = False
    def __init__(self, refm, disc_rate, learning_rate, starting_epsilon, batch_size):
        Agent.__init__(self, refm, disc_rate)
        self.ref_machine = refm
        self.num_states = refm.getNumObs()  # assuming that states = observations
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells = refm.getNumObsCells()
        self.state_vec_size = self.obs_cells*self.obs_symbols

        self.learning_rate = learning_rate
        self.starting_epsilon = starting_epsilon
        self.epsilon = starting_epsilon
        self.batch_size = math.floor(batch_size)
        self.criterion = get_criterion()

        self.prev_state = None
        self.prev_action = None
        self.TERMINAL_TOKEN = "TERMINAL"
        self.steps_done = 0



    def __str__(self):
        return "DeelQL()"

    def reset(self):
        # Replay buffer
        self.memory = ReplayMemory(10000)
        # Learning network
        self.policy_net = NeuralNet(self.state_vec_size, self.num_actions)
        self.previous_net = NeuralNet(self.state_vec_size, self.num_actions)
        self.optimizer = get_optimizer(self.policy_net, learning_rate=self.learning_rate)
        self.steps_done = 0
        self.epsilon = self.starting_epsilon


    def perceive(self, observations, reward):
        new_state_tensor = self.transferObservationToStateVec(observations)
        new_state_unsqueezed = new_state_tensor.unsqueeze(0)

        # Add to replay memory
        if(self.prev_state is not None) and (self.prev_action is not None):
            self.memory.push(
                self.prev_state,
                self.prev_action,
                new_state_unsqueezed,
                torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
            )

        # Do learning logic
        self.learn_from_experience()
        if self.steps_done % self.UPDATE_INTERVAL_LENGTH == 0:
            prev_net_weigths = self.previous_net.state_dict()
            policy_net_weights = self.previous_net.state_dict()
            for key in policy_net_weights:
                prev_net_weigths[key] = policy_net_weights[key]

        # Update epsilon
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon -= self.EPSILON_LINEAR_DECAY

        # Get action
        opt_action = self.getAction(new_state_tensor)

        # Cache current state and selected action
        self.prev_action = torch.tensor(opt_action).unsqueeze(0).unsqueeze(0)
        self.prev_state = new_state_unsqueezed
        self.steps_done += 1
        return opt_action

    def getQValue(self, state, action):
        values = self.previous_net.forward(state)
        return values[action].item()

    def transferObservationToStateVec(self, observations):
        if len(observations) != self.obs_cells:
            raise Exception("Observation is not in count as observation cells.")
        state_vec = torch.zeros(self.obs_cells*self.obs_symbols, dtype=torch.float32)
        for i in range(len(observations)):
            state_vec[observations[i]+i*self.obs_symbols] = 1
        return state_vec


    def computeActionFromQValue(self, state):
        action_values = [self.getQValue(state, action) for action in range(self.num_actions)]
        policy = np.argmax(action_values)
        return policy

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

    def learn_from_experience(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch
        # This converts batch-array of Transitions to Transition of batch-arrays
        batch = Transition(*zip(*transitions))

        # TO-DO Compute non-states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not self.TERMINAL_TOKEN, batch.next_state)),
            dtype=torch.bool)
        non_final_next_states = torch.cat(
            [state for state in batch.next_state if state is not self.TERMINAL_TOKEN]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q value
        # The model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net.
        # Selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected state value or 0 in case of
        # terminal state.
        next_state_values = reward_batch
        with torch.no_grad():
            next_state_values[non_final_mask] = next_state_values[non_final_mask] + self.disc_rate * \
                                                self.previous_net(non_final_next_states).max(1)[0]

        # Compute loss
        loss = self.criterion(state_action_values, next_state_values.unsqueeze(1))
        if self.LOG_LOSS:

            print("loss", loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

