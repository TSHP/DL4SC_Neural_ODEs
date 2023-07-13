import torch
from torch.nn.modules import Module


class KLDivLoss(Module):
    r"""Computes the KL-divergence loss for VAE training.

    Forward pass args:
        - mu (tensor): Latent mean.
        - log_var: Latent variance.
    """

    def __init__(self) -> None:
        super(KLDivLoss, self).__init__()

    def forward(self, mu: torch.tensor, log_sigma: torch.tensor) -> torch.tensor:
        return kl_divergence(mu, log_sigma)


def kl_divergence(mu: torch.tensor, log_var: torch.tensor):
    return torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )
