import torch
import torch.nn as nn
from torchdiffeq import odeint

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ODENet(nn.Module):
    def __init__(self, in_dim, out_channels):
        super(ODENet, self).__init__()

        w = 128
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=w, kernel_size=3, stride=1),
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

        self.rb = ODEBlock(ODEfunc(w))

        self.final = nn.Sequential(nn.Conv2d(in_channels=w, out_channels=out_channels, kernel_size=1, stride=1),
                      nn.BatchNorm2d(out_channels),
                      nn.Sigmoid())

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        out = self.downsample(x)
        out = self.rb(out)
        out = self.final(out)
        return out
