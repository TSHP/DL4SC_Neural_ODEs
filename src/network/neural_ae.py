import torch
import torch.nn as nn

from src.network.model_utils import norm
from src.network.node_utils import ODEBlock, ODEfunc

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
        ])

        self.upsampling_layer = nn.Sequential(*[
            nn.ConvTranspose2d(64, 64, 4, 2, 1, 0),
            norm(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, 2, 0, 0),
            norm(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 3, 1, 0, 0)
        ])

        self.enc = ODEBlock(ODEfunc(64))

        self.fc_mu = nn.Linear(64 * 36, self.latent_dim)
        self.fc_var = nn.Linear(64 * 36, self.latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim, 64 * 36)

        #self.dec = ODEBlock(ODEfunc(64, transpose=True))

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

        out = self.enc(out, torch.tensor([0, 1]).float().to(self.device))
        out = torch.flatten(out, start_dim=1)

        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        z = self.reparameterize(mu, log_var)

        out = self.decoder_input(z)
        out = out.view(-1, 64, 6, 6)
        out = self.enc(out, torch.tensor([1, 0]).float().to(self.device))

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
        out = self.enc(out, torch.tensor([1, 0]).float().to(self.device))

        samples = self.upsampling_layer(out)

        return samples