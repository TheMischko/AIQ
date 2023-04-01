import math
import random
import numpy as np
import torch

from agents.Agent import Agent
from agents.deep_ql.dql_config import DQLConfig
from agents.deep_ql.neural_utils.neuralNet import NeuralNet
from agents.deep_ql.neural_utils.neuralNet import get_optimizer
from agents.deep_ql.neural_utils.neuralNet import get_criterion
from agents.deep_ql.neural_utils.replayMemory import ReplayMemory, Transition
from agents.deep_ql.neural_utils.plottingTools import PlottingTools


class IDeepQLAgent(Agent):
    START_EPSILON = DQLConfig["start_epsilon"]
    MIN_EPSILON = DQLConfig["min_epsilon"]
    REWARD_DIVIDER = DQLConfig["reward_divider"]
    PLOT_Q_VALUES = DQLConfig["plot_Q_values"]
    PLOT_LOSS = DQLConfig["plot_rewards"]
    PLOT_ACTIONS_TAKEN = DQLConfig["plot_actions_taken"]
    PLOT_REWARDS = DQLConfig["plot_rewards"]
    PLOT_PROBABILITY = DQLConfig["plot_probability"]
    STATE_FOR_Q_VALUES_SAVING = DQLConfig["state_for_Q_values_saving"]

    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon_decay_length, neural_size_l1,
                 neural_size_l2, neural_size_l3):
        Agent.__init__(self, refm, disc_rate)
        self.optimizer = None
        self.neural_size_l1 = neural_size_l1
        self.neural_size_l2 = neural_size_l2
        self.neural_size_l3 = neural_size_l3
        self.policy_net = None
        self.target_net = None
        self.memory = None
        self.ref_machine = refm
        self.num_states = refm.getNumObs()  # assuming that states = observations
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells = refm.getNumObsCells()
        self.state_vec_size = self.obs_cells * self.obs_symbols
        self.neural_input_size = self.state_vec_size * 2
        self.gamma = gamma

        self.learning_rate = learning_rate
        self.starting_epsilon = self.START_EPSILON
        self.epsilon = self.START_EPSILON
        self.episodes_till_min_decay = epsilon_decay_length
        self.batch_size = math.floor(batch_size)
        self.criterion = get_criterion()

        self.cached_state_raw = None
        self.prev_state = None
        self.prev_action = None
        self.steps_done = 0

        # Plotting data
        self.last_losses = list()
        self.q_values_arr = [[] for i in range(self.num_actions)]
        self.actions_taken = list()
        self.rewards_given = list()

        self.plotting_tools = PlottingTools()
        self.epsilon_linear_decay = 0 if self.episodes_till_min_decay == 0 \
            else (self.starting_epsilon - self.MIN_EPSILON) / self.episodes_till_min_decay

    def reset(self):
        self.memory = ReplayMemory(10000)
        # Network evaluating Q function
        self.target_net = NeuralNet(self.neural_input_size, self.num_actions,
                                    self.neural_size_l1, self.neural_size_l2, self.neural_size_l3)
        # Network that is learning from replay memory
        self.optimizer = get_optimizer(self.target_net, learning_rate=self.learning_rate)
        self.steps_done = 0
        self.epsilon = self.starting_epsilon
        self.reset_values_for_plots()

    def perceive(self, observations, reward):
        new_state_tensor = self.transfer_observation_to_state_vec(observations)
        new_state_unsqueezed = new_state_tensor.unsqueeze(0)
        # Add to replay memory
        if (self.prev_state is not None) and (self.prev_action is not None):
            self.memory.push(
                self.prev_state,
                self.prev_action,
                new_state_unsqueezed,
                torch.tensor(reward / self.REWARD_DIVIDER, dtype=torch.float32).unsqueeze(0)
            )
        self.rewards_given.append(reward / self.REWARD_DIVIDER)

        # Do learning logic
        self.learn_from_experience()

        # Get action
        opt_action = self.get_action(new_state_tensor)

        # Cache current state and selected action
        self.prev_action = torch.tensor(opt_action).unsqueeze(0).unsqueeze(0)
        self.cached_state_raw = observations
        self.prev_state = new_state_unsqueezed

        return opt_action

    def episode_ended(self, stratum, program):
        if len(self.q_values_arr[0]) < 15:
            return
        if random.random() > self.PLOT_PROBABILITY:
            return

        if self.PLOT_LOSS and len(self.last_losses) > 0:
            self.plotting_tools.plot_array(y=self.last_losses, title=self.get_plot_title(stratum, program, "Loss"))

        if self.PLOT_Q_VALUES and len(self.q_values_arr[0]) > 0:
            self.plotting_tools.plot_multiple_array(self.q_values_arr,
                                                    title=self.get_plot_title(stratum, program, "Q-values"))

        if self.PLOT_REWARDS and len(self.rewards_given) > 0:
            self.plotting_tools.plot_array(y=self.rewards_given,
                                           title=self.get_plot_title(stratum, program, "Rewards"), type="o")

        if self.PLOT_ACTIONS_TAKEN and len(self.actions_taken) > 0:
            self.plotting_tools.plot_array(y=self.actions_taken,
                                           title=self.get_plot_title(stratum, program, "Actions taken"), type="o")

    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, random action will be taken or
          otherwise the best policy action will be taken.
        """
        is_random = random.random() < self.epsilon
        legal_actions = [action for action in range(self.num_actions)]
        action = None
        if is_random:
            action = random.choice(legal_actions)
        else:
            action = self.compute_action_from_Q_value(state)

        if self.PLOT_ACTIONS_TAKEN:
            self.actions_taken.append(action)

        return action

    def compute_action_from_Q_value(self, state):
        with torch.no_grad():
            action_values = self.target_net.forward(state).tolist()

            if self.PLOT_Q_VALUES:
                self.append_q_values(action_values, state)

            policy = np.argmax(action_values)
            return policy

    def transfer_observation_to_state_vec(self, observations):
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
        if self.episodes_till_min_decay == 0:
            return
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon -= self.epsilon_linear_decay
        else:
            self.epsilon = self.MIN_EPSILON

    def learn_from_experience(self):
        """
        Function that should handle learning of agent's neural net.
        """
        raise NotImplementedError()

    def reset_values_for_plots(self):
        self.last_losses = list()
        self.actions_taken = list()
        self.rewards_given = list()
        self.q_values_arr = [[] for i in range(self.num_actions)]

    def append_q_values(self, q_values, state):
        for i, state_ref_i in enumerate(self.STATE_FOR_Q_VALUES_SAVING):
            state_i = int(state[i].item())
            if state_ref_i is not state_i:
                return
        for i, q_value in enumerate(q_values):
            self.q_values_arr[i].append(q_value)

    def get_plot_title(self, stratum, program, title):
        return "%s for: S%s\n%s" % (title, stratum, program)
