import torch
import torch.nn as nn

from src.network.utils.model import ResBlock, upsample


class ResUNet(nn.Module):
    def __init__(self, num_filters, kernel_size, out_dim):
        super(ResUNet, self).__init__()

        # Encoder
        initial = ResBlock(
            [num_filters[0], num_filters[1]], kernel_size, strides=(1, 1), first_layer=True
        )
        self.encoder = nn.Sequential(
            *(
                [initial]
                + [
                    ResBlock(
                        [num_filters[i], num_filters[i + 1]],
                        kernel_size,
                        strides=(2, 2),
                    )
                    for i in range(1, len(num_filters) - 1)
                ]
            )
        )

        # Bridge
        self.bridge = ResBlock(
            [num_filters[-1], num_filters[-1]], kernel_size, strides=(1, 1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            *(
                [
                    ResBlock(
                        [num_filters[-i] * 2, num_filters[-i-1]],
                        kernel_size,
                        strides=(1, 1),
                    )
                    for i in range(1, len(num_filters) - 1)
                ]
            )
        )

        self.output = nn.Conv2d(num_filters[1], out_dim, kernel_size=1, stride=1)

    def forward(self, x):
        skip_connections = []
        out = x
        for layer in self.encoder:
            out = layer(out)
            skip_connections.append(out)

        out = self.bridge(out)

        for layer, skip_features in zip(self.decoder, reversed(skip_connections)):
            out = upsample(out, skip_features.size()[2:])
            out = torch.cat([out, skip_features], dim=1)
            out = layer(out)

        out = upsample(out, (256, 256))  # Read from params

        return self.output(out)
