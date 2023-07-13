import torch
import torch.nn as nn

from src.network.utils.model import norm
from src.network.utils.node import ODEBlock, ODEfunc


class ODENet(nn.Module):
    def __init__(self, in_dim, out_channels):
        super(ODENet, self).__init__()

        w = 128
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=w, kernel_size=3, stride=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=w, out_channels=w, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=w, out_channels=w, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=w, out_channels=w, kernel_size=3, stride=2, padding=1
            ),
        )

        self.rb = ODEBlock(ODEfunc(w))

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=w, out_channels=20, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        out = self.downsample(x)
        out = self.rb(out)
        out = self.final(out)
        return out
