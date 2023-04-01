import math

import torch

from agents.deep_ql.DQL_Dual_Decay import DQL_Dual_Decay
from agents.deep_ql.neural_utils import traces
from agents.deep_ql.neural_utils.neuralNet import get_criterion


class DQL_Dual_ET_Decay(DQL_Dual_Decay):

    TRACES_METHODS = [
        "replacing",
        "accumulating",
        "dutch"
    ]

    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon_decay_length, neural_size_l1,
                 neural_size_l2, neural_size_l3, tau, update_interval_length, lambda_val, traces_method):
        super().__init__(refm, disc_rate, learning_rate, gamma, batch_size, epsilon_decay_length, neural_size_l1,
                         neural_size_l2, neural_size_l3, tau, update_interval_length)
        self.eligibility = torch.zeros((self.batch_size, self.num_actions))
        self.lambda_val = lambda_val
        self.trace_method = self.TRACES_METHODS[int(traces_method)]
        self.criterion = get_criterion(reduction='none')

    def reset(self):
        super().reset()
        self.eligibility = torch.zeros(self.state_vec_size, self.state_vec_size, self.num_actions)

    def learn_from_experience(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch = self.get_learning_batches()

        q_values = self.policy_net(state_batch)
        q_values = q_values.gather(1, action_batch).squeeze()

        q_next_values = None
        with torch.no_grad():
            q_next_values = reward_batch + self.gamma * self.target_net(next_state_batch).max(1)[0]

        td_error = self.criterion(q_values, q_next_values)

        # update eligibility traces
        if self.trace_method is not None:
            if self.steps_done == 0:
                self.eligibility = self.reset_trace()
            self.eligibility = self.update_trace(action_batch.squeeze())
            el_vals = self.eligibility.gather(1, action_batch).squeeze(1)
            td_error *= el_vals
            self.eligibility *= self.gamma*self.lambda_val

        # td update
        td_error = td_error.mean()
        self.optimizer.zero_grad()
        td_error.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        self.last_losses.append(td_error.detach().item())

        # Update epsilon
        self.decrement_epsilon()

        if self.steps_done % self.update_interval_length == 0:
            self.copy_network_weights()
        self.steps_done += 1

    def update_trace(self, actions):
        return getattr(traces, self.trace_method)(self.learning_rate, actions, self.eligibility)

    def reset_trace(self):
        return torch.zeros((self.batch_size, self.num_actions))

    def __str__(self):
        return "DQL_Dual_Decay(%.4f,%.2f,%d,%d,%d,%d,%d,%.2f,%d,%.2f,%s)" % (
            self.learning_rate,
            self.gamma,
            self.batch_size,
            self.episodes_till_min_decay,
            self.neural_size_l1,
            self.neural_size_l2,
            self.neural_size_l3,
            self.tau,
            self.update_interval_length,
            self.lambda_val,
            self.trace_method
        )



