import torch
from torch import nn

from agents.neural_utils.neuralNet import NeuralNet


class DistributionNeuralNet(nn.Module):
    def __init__(self, input_size, l1_size, l2_size, l3_size, num_classes, num_atoms):
        self.has_fourth_layer = l3_size > 0
        super().__init__()
        self.num_atoms = num_atoms
        self.classes = num_classes
        self.output_size = self.num_atoms * self.classes
        self.l1 = nn.Linear(int(input_size), int(l1_size))
        self.l2 = nn.Linear(int(l1_size), int(l2_size))
        if self.has_fourth_layer:
            self.l3 = nn.Linear(int(l2_size), int(l3_size))
            self.l4 = nn.Linear(int(l3_size), int(self.output_size))
            nn.init.zeros_(self.l4.weight)
        else:
            self.l3 = nn.Linear(int(l2_size), self.output_size)
        nn.init.zeros_(self.l1.weight)
        nn.init.zeros_(self.l2.weight)
        nn.init.zeros_(self.l3.weight)

    def forward(self, x):
        x_dims = x.size()
        batch_size = x_dims[0] if x.ndim > 1 else 0
        x = x.type(torch.float32)
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        if self.has_fourth_layer:
            x = torch.sigmoid(self.l3(x))
            x = self.l4(x)
        else:
            x = self.l3(x)
        x = torch.relu(x)
        x = self.split_into_distributions(x, batch_size)
        x = torch.softmax(x, dim=1)
        x = self.inverse_split(x, batch_size)
        return x

    def split_into_distributions(self, x, batch_size):
        if batch_size > 0:
            return x.reshape(batch_size, self.classes, self.num_atoms)
        return x.reshape(self.classes, self.num_atoms)

    def inverse_split(self, x, batch_size):
        if batch_size > 0:
            return x.reshape(batch_size, self.classes * self.num_atoms)
        return x.reshape(self.classes * self.num_atoms)