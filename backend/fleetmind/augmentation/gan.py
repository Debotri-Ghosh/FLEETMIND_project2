"""Conditional WaveGAN-lite for bearing vibration windows (Phase 3).

Design (CPU-friendly, M.Tech-demo scale):
- Operates on DOWNSAMPLED windows (2048 -> 256 by mean-pooling x8). The
  augmentation script upsamples synthetic windows back to 2048 with linear
  interpolation before feeding them to BearingCNN.
- Conditional on fault class via a learned label embedding concatenated to
  the noise vector (generator) and to the waveform (discriminator).
- WGAN objective with weight clipping (stable on CPU, no gradient-penalty
  cost): D maximizes E[D(real)] - E[D(fake)], G maximizes E[D(fake)].

Shapes: noise (B, latent_dim), label (B,) -> synthetic (B, 1, 256).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_classes: int = 4, latent_dim: int = 64, out_len: int = 256):
        super().__init__()
        self.out_len = out_len
        self.label_emb = nn.Embedding(num_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_len),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self.label_emb(labels)], dim=1)
        return self.net(h).unsqueeze(1)  # (B, 1, out_len)


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 4, in_len: int = 256):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(in_len + 16, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x.squeeze(1), self.label_emb(labels)], dim=1)
        return self.net(h).squeeze(1)  # (B,)


class ConditionalWaveGAN:
    """WGAN with weight clipping; train() runs the alternating updates."""

    def __init__(
        self,
        num_classes: int = 4,
        latent_dim: int = 64,
        seq_len: int = 256,
        lr: float = 1e-4,
        clip: float = 0.01,
        n_critic: int = 5,
        lr_g: float | None = None,
        device: str = "cpu",
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.clip = clip
        self.gen = Generator(num_classes, latent_dim, seq_len).to(device)
        self.disc = Discriminator(num_classes, seq_len).to(device)
        # TTUR: generator learns 2x faster than critic (stabilizes class
        # conditioning and high-frequency learning on small CPU models).
        self.opt_g = torch.optim.RMSprop(self.gen.parameters(), lr=(lr_g or lr * 2))
        self.opt_d = torch.optim.RMSprop(self.disc.parameters(), lr=lr)

    def train_step(self, real: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        real, labels = real.to(self.device), labels.to(self.device)
        b = real.size(0)
        # --- critic ---
        for _ in range(self.n_critic):
            self.opt_d.zero_grad()
            with torch.no_grad():
                z = torch.randn(b, self.latent_dim, device=self.device)
                fake = self.gen(z, labels)
            loss_d = -(self.disc(real, labels).mean() - self.disc(fake, labels).mean())
            loss_d.backward()
            self.opt_d.step()
            for p in self.disc.parameters():
                p.data.clamp_(-self.clip, self.clip)
        # --- generator ---
        self.opt_g.zero_grad()
        z = torch.randn(b, self.latent_dim, device=self.device)
        fake = self.gen(z, labels)
        loss_g = -self.disc(fake, labels).mean()
        loss_g.backward()
        self.opt_g.step()
        return {"loss_d": loss_d.item(), "loss_g": loss_g.item()}

    @torch.no_grad()
    def generate(self, labels: list[int]) -> torch.Tensor:
        self.gen.eval()
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        z = torch.randn(len(labels), self.latent_dim, device=self.device)
        out = self.gen(z, y).cpu()
        self.gen.train()
        return out
