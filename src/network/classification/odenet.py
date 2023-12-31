import torch
import torch.nn as nn

from src.network.utils.node import AdjointODEBlock, ODEBlock, ODEfunc


class ODENet(nn.Module):
    def __init__(self, out_dim, adjoint=False, rtol=1e-7, atol=1e-9, method="dopri5"):
        super(ODENet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.downsampling_layer = nn.Sequential(
            *[
                nn.Conv2d(1, 64, 3, 1),
                self.bn,
                self.relu,
                nn.Conv2d(64, 64, 4, 2, 1),
                self.bn,
                self.relu,
                nn.Conv2d(64, 64, 4, 2, 1),
            ]
        )

        self.node = ODEBlock(ODEfunc(64), rtol=rtol, atol=atol, method=method) if not adjoint else AdjointODEBlock(ODEfunc(64), rtol=rtol, atol=atol, method=method)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        out = self.downsampling_layer(x)
        out = self.node(out, torch.tensor([1, 0]).float().to(self.device))

        out = self.bn(out)
        out = self.relu(out)
        out = self.adaptive_pool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out
    
    def set_solver(self, adjoint=False, rtol=1e-7, atol=1e-9, method="dopri5"):
        self.node.set_solver(adjoint=adjoint, rtol=rtol, atol=atol, method=method)
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
