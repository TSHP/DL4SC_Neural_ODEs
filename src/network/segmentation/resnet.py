import torch.nn as nn

from src.network.utils.model import ResBlock


# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=6):
        super(ResNet, self).__init__()
        w = 128
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=w, kernel_size=3, stride=1),
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
        self.res_blocks = nn.Sequential(
            *[ResBlock(64, 64) for _ in range(num_res_blocks)]
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=w, out_channels=out_channels, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.downsample(x)
        out = self.residual_blocks(out)
        out = self.final(out)
        return out
