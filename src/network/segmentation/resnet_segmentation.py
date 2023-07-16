import torch.nn as nn

from src.network.utils.model import ResBlock

# Define the ResNet model
class SegmentationResNet(nn.Module):
    def __init__(self, num_filters, kernel_size, out_dim):
        super(SegmentationResNet, self).__init__()
        w = num_filters[1]
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=num_filters[0], out_channels=w, kernel_size=3, stride=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=w, out_channels=w, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=w, out_channels=w, kernel_size=3, stride=2, padding=1),
        )
        self.residual_blocks = nn.Sequential(
            ResBlock([w, w], kernel_size, strides=(1, 1)),
            ResBlock([w, w], kernel_size, strides=(1, 1)),
            ResBlock([w, w], kernel_size, strides=(1, 1)),
            ResBlock([w, w], kernel_size, strides=(1, 1)),
            ResBlock([w, w], kernel_size, strides=(1, 1)),
            ResBlock([w, w], kernel_size, strides=(1, 1))
        )
        self.upsample = nn.Sequential(nn.ConvTranspose2d(w, w, kernel_size=2, stride=2, padding=0),
                                   nn.BatchNorm2d(w),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(w, w, kernel_size=2, stride=2, padding=0),
                                   nn.BatchNorm2d(w),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=w, out_channels=out_dim, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_dim),
                                   nn.Sigmoid())

    def forward(self, x):
        out = self.downsample(x)
        out = self.residual_blocks(out)
        out = self.upsample(out)
        return out
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
