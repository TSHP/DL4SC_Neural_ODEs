import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super(MLP, self).__init__()
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.input_layer = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)]
        )
        self.output_layer = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        x = self.flatten(x)
        out = self.input_layer(x)
        out = self.relu(out)

        for layer in enumerate(self.hidden_layers):
            out = layer(out)
            out = self.relu(out)
        
        out = self.output_layer(out)
        out = self.relu(out)
        return out
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
