import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.Relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.Relu(out)
        return out