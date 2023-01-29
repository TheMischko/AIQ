import numpy as np
import torch

from agents.neural_utils.IDeepQLAgent import IDeepQLAgent


class VanillaDeepQL(IDeepQLAgent):
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

    def __str__(self):
        return "Vanilla DeepQL"
