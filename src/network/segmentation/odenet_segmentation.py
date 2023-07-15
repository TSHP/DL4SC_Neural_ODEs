import torch
import torch.nn as nn
from src.network.utils.node import ODEBlock, AdjointODEBlock, ODEfunc

class SegmentationODENet(nn.Module):
    def __init__(self, num_filters, kernel_size, out_dim, adjoint=False, rtol=1e-7, atol=1e-9, method="dopri5"):
        super(SegmentationODENet, self).__init__()

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

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

        self.rb = ODEBlock(ODEfunc(w), rtol=rtol, atol=atol, method=method) if not adjoint else AdjointODEBlock(ODEfunc(w), rtol=rtol, atol=atol, method=method)

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
        out = self.rb(out, torch.tensor([1, 0]).float().to(self.device))
        out = self.upsample(out)
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
