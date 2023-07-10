import torch.nn as nn

from src.network.model_utils import norm

class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        return out

class RungeKuttaIntegrator(nn.Module):
    def __init__(self, f, dt):
        super(RungeKuttaIntegrator, self).__init__()
        self.f = f
        self.dt = dt

    def forward(self, x):
        k1 = self.f(x)
        k2 = self.f(x + self.dt * k1 / 2)
        k3 = self.f(x + self.dt * k2 / 2)
        k4 = self.f(x + self.dt * k3)
        return x + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    

class RKNet(nn.Module):
    def __init__(self, out_dim):
        super(RKNet, self).__init__()
        self.downsampling_layer = nn.Sequential(*[
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1)
        ])

        self.rb = RungeKuttaIntegrator(ODEfunc(64), 0.001)

        self.flatten = nn.Flatten()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.norm = norm(64)
        self.fc = nn.Linear(64, out_dim)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        out = self.downsampling_layer(x)
        out = self.rb(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.adaptive_pool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out