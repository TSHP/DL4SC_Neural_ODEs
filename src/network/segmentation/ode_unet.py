import torch
import torch.nn as nn

from src.network.utils.model import InitialResBlock, upsample
from src.network.utils.node import ODEfunc, ODEBlock


class ODEUNet(nn.Module):
    def __init__(self, num_filters, kernel_size, out_dim, adjoint=False, rtol=1e-7, atol=1e-9, method="dopri5"):
        super(ODEUNet, self).__init__()

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        # Encoder
        self.initial = InitialResBlock(
            [num_filters[0], num_filters[1]], kernel_size, strides=(1, 1)
        )
        self.encoder = nn.Sequential(
            *([
                    ODEBlock(ODEfunc(num_filters[i]), adjoint=adjoint, rtol=rtol, atol=atol, method=method)
                    for i in range(1, len(num_filters) - 1)
                ]
            )
        )

        self.downsamplers = nn.Sequential(*[nn.Conv2d(num_filters[i], num_filters[i+1], kernel_size=3, padding=1, stride=2) for i in range(1, len(num_filters) - 1)])

        # Bridge
        self.bridge = ODEBlock(ODEfunc(num_filters[-1]), adjoint=adjoint, rtol=rtol, atol=atol, method=method)

        # Decoder
        self.decoder = nn.Sequential(
            *(
                [
                    ODEBlock(ODEfunc(num_filters[-i] * 2), adjoint=adjoint, rtol=rtol, atol=atol, method=method)
                    for i in range(1, len(num_filters) - 1)
                ]
            )
        )

        self.upsamplers = nn.Sequential(*[nn.Conv2d(num_filters[-i] * 2, num_filters[-i-1], kernel_size=3, padding=1) for i in range(1, len(num_filters) - 1)])

        self.output = nn.Conv2d(num_filters[1], out_dim, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.initial(x)
        skip_connections = [out]
        for layer, downsampler in zip(self.encoder, self.downsamplers):
            out = layer(out, torch.tensor([1, 0]).float().to(self.device))
            skip_connections.append(out)
            out = downsampler(out)

        out = self.bridge(out, torch.tensor([1, 0]).float().to(self.device))

        for layer, skip_features, upsampler in zip(self.decoder, reversed(skip_connections), self.upsamplers):
            out = upsample(out, skip_features.size()[2:])
            out = torch.cat([out, skip_features], dim=1)
            out = layer(out, torch.tensor([1, 0]).float().to(self.device))
            out = upsampler(out)

        out = upsample(out, (256, 256))  # Read from params

        return self.output(out)
