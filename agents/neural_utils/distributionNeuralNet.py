import torch
from torch import nn

from agents.neural_utils.neuralNet import NeuralNet


class DistributionNeuralNet(nn.Module):
    def __init__(self, input_size, num_classes, num_atoms, hidden_size):
        super(DistributionNeuralNet, self).__init__()

        self.num_atoms = num_atoms
        self.classes = num_classes
        self.output_size = self.num_atoms * self.classes
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, self.output_size)

        # Weight initialization
        nn.init.kaiming_uniform_(self.l1.weight)
        nn.init.kaiming_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        x = torch.softmax(x.view(-1, self.num_atoms), dim=1)
        return x.view(-1, self.classes, self.num_atoms)
