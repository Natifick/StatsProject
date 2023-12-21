import torch
import torch.nn as nn
import numpy as np

class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dim=None):
        super().__init__()
        self.register_buffer('in_dim', torch.tensor([in_dim]))
        self.register_buffer('latent_dim', torch.tensor([latent_dim]))
        if hidden_dim is None:
            hidden_dim = in_dim
        self.register_buffer('hidden_dim', torch.tensor([hidden_dim]))

        self._init_submodels()
    
    def _init_submodels(self):
        self.encoder = Encoder(self.in_dim.item(), self.latent_dim.item(), self.hidden_dim.item())
        self.decoder = Decoder(self.latent_dim.item(), self.in_dim.item(), self.hidden_dim.item())
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def prior(self, n):
        return torch.randn(n, self.latent_dim, device=self.device)
    
    def forward(self, x):
        mu_z, log_std_z = self.encoder(x)
        z = self.prior(x.shape[0]) * log_std_z.exp() + mu_z
        x_recon = self.decoder(z)
        return mu_z, log_std_z, x_recon
    
    def loss(self, x):
        mu_z, log_std_z, x_recon = self(x)
        recon_loss = get_normal_nll(x, x_recon, torch.zeros_like(x)).sum(1).mean()
        kl_loss = get_normal_kl(mu_z, log_std_z).sum(1).mean()
        return kl_loss + recon_loss

    @torch.no_grad()
    def sample(self, n):
        z = self.prior(n)
        return self.decoder(z)
    
    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)[0]


class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )
    
    def forward(self, x):
        mu, log_std = self.net(x).chunk(2, dim=1)
        return mu, log_std


class Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, z):
        return self.net(z)


def get_normal_nll(x, mean, log_std):
    return (
        0.5 * np.log(2 * np.pi)
        + log_std
        + (x - mean) ** 2 * np.exp(-2 * log_std) * 0.5
    )


def get_normal_kl(mu_1, log_std_1, mu_2=None, log_std_2=None):
    if mu_2 is None:
        mu_2 = torch.zeros_like(mu_1)
    if log_std_2 is None:
        log_std_2 = torch.zeros_like(log_std_1)

    return (
        (log_std_2 - log_std_1)
        + (torch.exp(log_std_1 * 2) + (mu_1 - mu_2) ** 2)
        / 2
        / torch.exp(log_std_2 * 2)
        - 0.5
    )

