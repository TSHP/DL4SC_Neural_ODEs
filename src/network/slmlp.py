import torch.nn as nn


class SLMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(SLMLP, self).__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.Relu = nn.ReLU()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        out = self.input_layer(x)
        out = self.Relu(out)
        out = self.linear(out)
        out = self.Relu(out)
        return out