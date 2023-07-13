import torch
import torch.nn as nn


class SLMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(SLMLP, self).__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x)
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.linear(out)
        out = self.relu(out)
        return out
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
