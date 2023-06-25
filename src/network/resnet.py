import torch.nn as nn

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut

class ResNet(nn.Module):
    def __init__(self, out_dim):
        super(ResNet, self).__init__()

        self.downsampling_layer = nn.Sequential(*[
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1)
        ])

        self.rb1 = ResBlock(64, 64)
        self.rb2 = ResBlock(64, 64)
        self.rb3 = ResBlock(64, 64)
        self.rb4 = ResBlock(64, 64)
        self.rb5 = ResBlock(64, 64)
        self.rb6 = ResBlock(64, 64)

        self.flatten = nn.Flatten()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.norm = norm(64)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        out = self.downsampling_layer(x)
        out = self.rb1(out)
        out = self.rb2(out)
        out = self.rb3(out)
        out = self.rb4(out)
        out = self.rb5(out)
        out = self.rb6(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.adaptive_pool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out