import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_filters, kernel_size, strides, first_layer=False):
        super(ResBlock, self).__init__()

        self.first_layer = first_layer
        self.num_filters = num_filters
        self.strides = strides

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        padding = self.calculate_padding(kernel_size)
        self.norm1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_filters[0], num_filters[0], kernel_size, strides, padding=padding
        )
        self.norm2 = nn.BatchNorm2d(num_filters[0])
        self.conv2 = nn.Conv2d(
            num_filters[0], num_filters[1], kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = x
        if not self.first_layer:
            out = self.norm1(x)
            out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.num_filters[0] != self.num_filters[1]:
            x = nn.Conv2d(
                self.num_filters[0], self.num_filters[1], kernel_size=1, stride=self.strides, device=self.device)(x)
            x = nn.BatchNorm2d(self.num_filters[1], device=self.device)(x)

        return out + x

    def calculate_padding(self, kernel_size):
        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        return (pad_left, pad_left)
    

def upsample(x, target_size):
        x_resized = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x_resized
