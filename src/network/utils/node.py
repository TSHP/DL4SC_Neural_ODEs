from functools import partial

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint

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


class ODEfunc(nn.Module):
    def __init__(self, dim, transpose=False):
        super(ODEfunc, self).__init__()
        self.norm1 = nn.GroupNorm(min(32, dim), dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1, transpose=transpose)
        self.norm2 = nn.GroupNorm(min(32, dim), dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1, transpose=transpose)
        self.norm3 = nn.GroupNorm(min(32, dim), dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-7, atol=1e-9, method="dopri5"):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

        # Set solver
        self.set_solver(rtol=rtol, atol=atol, method=method)

    def forward(self, x, t):
        t = t.type_as(x)
        out = self.solver(self.odefunc, x, t)
        return out[1]

    def set_solver(self, rtol=1e-7, atol=1e-9, method="dopri5"):
        self.solver = partial(odeint, rtol=rtol, atol=atol, method=method)

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class AdjointODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-7, atol=1e-9, method="dopri5"):
        super(AdjointODEBlock, self).__init__()
        self.odefunc = odefunc

        # Set solver
        self.set_solver(rtol=rtol, atol=atol, method=method)

    def forward(self, x, t):
        t = t.type_as(x)
        out = self.solver(self.odefunc, x, t)
        return out[1]

    def set_solver(self, rtol=1e-7, atol=1e-9, method="dopri5"):
        self.solver = partial(odeint_adjoint, rtol=rtol, atol=atol, method=method)

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
