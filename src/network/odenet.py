import torch.nn as nn


class ODENet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ODENet, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.Relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.Relu(out)
        return out