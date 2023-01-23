import torch.optim
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 8)
        self.l4 = nn.Linear(8, output_size)
        nn.init.zeros_(self.l1.weight)
        nn.init.zeros_(self.l2.weight)
        nn.init.zeros_(self.l3.weight)

    def forward(self, x):
        x = x.type(torch.float32)
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return self.l4(x)


def get_optimizer(model, learning_rate=0.001):
    return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    #return torch.optim.AdamW(model.parameters(), learning_rate, amsgrad=True)


def get_criterion():
    return nn.SmoothL1Loss()
