import torch
import torch.nn as nn

from network.utils.model import norm
from network.utils.node import ODEBlock, ODEfunc


class NeuralVAE(nn.Module):
    def __init__(self, latent_dim, adjoint=False, rtol=1e-7, atol=1e-9, method="dopri5"):
        super(NeuralVAE, self).__init__()

        self.latent_dim = latent_dim

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.downsampling_layer = nn.Sequential(
            *[
                nn.Conv2d(1, 64, 3, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
            ]
        )

        self.upsampling_layer = nn.Sequential(
            *[
                nn.ConvTranspose2d(64, 64, 4, 2, 1, 0),
                norm(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 64, 4, 2, 0, 0),
                norm(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 1, 3, 1, 0, 0),
            ]
        )

        self.node = ODEBlock(ODEfunc(64), adjoint=adjoint, rtol=rtol, atol=atol, method=method)
        self.fc_mu = nn.Linear(64 * 36, self.latent_dim)
        self.fc_var = nn.Linear(64 * 36, self.latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim, 64 * 36)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encoder(self, x):
        out = self.downsampling_layer(x)

        # Solve the ODE forward
        out = self.node(out, torch.tensor([0, 1]).float().to(self.device))
        out = torch.flatten(out, start_dim=1)

        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        z = self.reparameterize(mu, log_var)

        return z, mu, log_var

    def decoder(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, 64, 6, 6)

        # Solve the ODE backwards
        out = self.node(out, torch.tensor([1, 0]).float().to(self.device))

        return self.upsampling_layer(out)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        out = self.decoder(z)
        return out, mu, log_var

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        return self.decoder(z)
    
    def set_solver(self, adjoint=False, rtol=1e-7, atol=1e-9, method="dopri5"):
        self.node.set_solver(adjoint=adjoint, rtol=rtol, atol=atol, method=method)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
