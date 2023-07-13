import torch
import torch.nn as nn

from network.utils.model import ResBlock, norm

class ResNet(nn.Module):
    def __init__(self, out_dim, num_res_blocks=6):
        super(ResNet, self).__init__()

        self.downsampling_layer = nn.Sequential(
            *[
                nn.Conv2d(1, 64, 3, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
            ]
        )

        self.res_blocks = nn.Sequential(*[ResBlock(64, 64) for _ in range(num_res_blocks)])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        out = self.downsampling_layer(x)
        out = self.res_blocks(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.adaptive_pool(out)
        out = torch.flatten(out)
        out = self.fc(out)
        return out
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)