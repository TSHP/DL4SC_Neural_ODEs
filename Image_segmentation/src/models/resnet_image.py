import torch
import torch.nn as nn
from torch.nn import ReLU
from torch import optim

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out += identity
        out = self.relu(out)

        return out

# Define the ResNet model
class ResNet6_images(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet6_images, self).__init__()
        w = 64
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=w, kernel_size=3, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),

        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(w, w),
            ResidualBlock(w, w),
            ResidualBlock(w, w),
            ResidualBlock(w, w),#
            ResidualBlock(w, w),
            ResidualBlock(w, w),
        )
        self.upsample = nn.Sequential(nn.Conv2d(in_channels=w, out_channels=out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        out = self.downsample(x)
        out = self.residual_blocks(out)
        out = self.upsample(out)
        return out