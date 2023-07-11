import torch
import torch.nn as nn
from torchdiffeq import odeint

from src.network.model_utils import norm, ConcatConv2d

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
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class NeuralAE(nn.Module):
    def __init__(self, latent_dim):
        super(NeuralAE, self).__init__()

        self.latent_dim = latent_dim

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.downsampling_layer = nn.Sequential(*[
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1)
        ]).to(self.device)

        self.upsampling_layer = nn.Sequential(*[
            nn.ConvTranspose2d(64, 64, 4, 2, 1, 0),
            norm(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, 2, 0, 0),
            norm(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 3, 1, 0, 0)
        ]).to(self.device)

        self.enc = ODEBlock(ODEfunc(64)).to(self.device)

        self.fc_mu = nn.Linear(64 * 36, self.latent_dim).to(self.device)
        self.fc_var = nn.Linear(64 * 36, self.latent_dim).to(self.device)
        self.decoder_input = nn.Linear(self.latent_dim, 64 * 36).to(self.device)

        self.dec = ODEBlock(ODEfunc(64, transpose=True)).to(self.device)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        out = self.downsampling_layer(x)

        out = self.enc(out)
        out = torch.flatten(out, start_dim=1)

        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        z = self.reparameterize(mu, log_var)

        out = self.decoder_input(z)
        out = out.view(-1, 64, 6, 6)
        out = self.dec(out)

        out = self.upsampling_layer(out)
        
        return out, mu, log_var
    
    def kl_loss(self, mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = torch.randn(num_samples, 10).to(self.device)

        out = self.decoder_input(z)
        out = out.view(-1, 64, 6, 6)
        out = self.dec(out)

        samples = self.upsampling_layer(out)

        return samples