from functools import partial

import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint

from network.utils.model import norm, ConcatConv2d

class ODEfunc(nn.Module):
    def __init__(self, dim, transpose=False):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1, transpose=transpose)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1, transpose=transpose)
        self.norm3 = norm(dim)
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
    def __init__(self, odefunc, adjoint=False, rtol=1e-7, atol=1e-9, method="dopri5"):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

        # Set solver
        self.set_solver(adjoint=adjoint, rtol=rtol, atol=atol, method=method)

    def forward(self, x, t):
        t = t.type_as(x)
        out = odeint(self.odefunc, x, t, rtol=1e-3, atol=1e-3)
        return out[1]
    
    def set_solver(self, adjoint=False, rtol=1e-7, atol=1e-9, method="dopri5"):
        if adjoint:
            self.solver = partial(odeint_adjoint, rtol=rtol, atol=atol, method=method)
        else:
            self.solver = partial(odeint, rtol=rtol, atol=atol, method=method)

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
