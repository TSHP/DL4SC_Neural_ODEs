import torch.nn as nn

from src.network.utils.model import ResBlock, norm


class ResNet(nn.Module):
    def __init__(
        self,
        num_res_blocks=6,
        num_filters=[64, 64],
        kernel_size=3,
        num_channels=1,
        out_dim=10,
    ):
        super(ResNet, self).__init__()

        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_channels = num_channels

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.norm = norm(64)

        self.downsampling_layer = nn.Sequential(
            *[
                nn.Conv2d(1, 64, 3, 1),
                self.norm,
                self.relu,
                nn.Conv2d(64, 64, 4, 2, 1),
                self.norm,
                self.relu,
                nn.Conv2d(64, 64, 4, 2, 1),
            ]
        )

        self.res_blocks = nn.Sequential(
            *[
                ResBlock(self.num_filters, self.kernel_size, strides=[1, 1])
                for _ in range(num_res_blocks)
            ]
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        out = self.downsampling_layer(x)
        out = self.res_blocks(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.adaptive_pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
