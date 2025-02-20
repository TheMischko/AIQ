#
# Trivial agent that takes random actions
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3
#
import torch

from agents.deep_ql.neural_utils.IDeepQLAgent import IDeepQLAgent
from agents.deep_ql.neural_utils.neuralNet import NeuralNet
from agents.deep_ql.neural_utils.neuralNet import get_optimizer


class DQL_Dual_Decay(IDeepQLAgent):
    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon_decay_length, neural_size_l1,
                 neural_size_l2, neural_size_l3, tau, update_interval_length):
        IDeepQLAgent.__init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon_decay_length,
                              neural_size_l1, neural_size_l2, neural_size_l3)
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
        return "DQL_Dual_Decay(%.4f,%.2f,%d,%d,%d,%d,%d,%.2f,%d)" % (
            self.learning_rate,
            self.gamma,
            self.batch_size,
            self.episodes_till_min_decay,
            self.neural_size_l1,
            self.neural_size_l2,
            self.neural_size_l3,
            self.tau,
            self.update_interval_length,
        )
