import torch.nn as nn

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

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
    def __init__(self, in_channels, out_channels=20):
        super(RKNet, self).__init__()
        w = 128
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=w, kernel_size=3, stride=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=w, out_channels=w, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=w, out_channels=w, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=w, out_channels=w, kernel_size=3, stride=2, padding=1),
        )

        self.rb = RungeKuttaIntegrator(ODEfunc(w), 0.001)

        self.final = nn.Sequential(nn.Conv2d(in_channels=w, out_channels=out_channels, kernel_size=1, stride=1),
                      nn.BatchNorm2d(out_channels),
                      nn.Sigmoid(inplace=True))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        out = self.downsample(x)
        out = self.rb(out)
        out = self.final(out)
        return out
