import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(x, target_size):
    x_resized = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
    return x_resized


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ConcatConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ResBlock(nn.Module):
    def __init__(self, num_filters, kernel_size, strides):
        super(ResBlock, self).__init__()

        self.num_filters = num_filters
        self.strides = strides

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
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        x = nn.Conv2d(
            self.num_filters[0], self.num_filters[1], kernel_size=1, stride=self.strides
        )(x)
        x = nn.BatchNorm2d(self.num_filters[1])(x)

        return out + x

    def calculate_padding(self, kernel_size):
        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        return (pad_left, pad_left)


class InitialResBlock(nn.Module):
    def __init__(self, num_filters, kernel_size, strides):
        super(InitialResBlock, self).__init__()

        self.num_filters = num_filters

        padding = self.calculate_padding(kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_filters[0], num_filters[0], kernel_size, strides, padding=padding
        )
        self.norm2 = nn.BatchNorm2d(num_filters[0])
        self.conv2 = nn.Conv2d(
            num_filters[0], num_filters[1], kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        x = nn.Conv2d(
            self.num_filters[0], self.num_filters[1], kernel_size=1, stride=1
        )(x)
        x = nn.BatchNorm2d(self.num_filters[1])(x)

        return out + x

    def calculate_padding(self, kernel_size):
        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        return (pad_left, pad_left)
