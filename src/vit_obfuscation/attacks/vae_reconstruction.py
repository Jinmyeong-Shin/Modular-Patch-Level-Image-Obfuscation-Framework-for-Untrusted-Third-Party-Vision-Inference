from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    """VAE conditioned on obfuscated images to reconstruct originals."""

    def __init__(self, image_size: int = 224, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: obfuscated image -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        # Decoder: latent + obfuscated features -> reconstruction
        self.decoder_input = nn.Linear(latent_dim, 256 * 14 * 14)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z).view(-1, 256, 14, 14)
        return self.decoder(h)

    def forward(
        self, obfuscated: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, H, W = obfuscated.shape
        mu, logvar = self.encode(obfuscated)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        # Resize to match input dimensions
        if recon.shape[-2:] != (H, W):
            recon = F.interpolate(
                recon, size=(H, W), mode="bilinear", align_corners=False
            )
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, target, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


def train_vae_attack(
    obfuscated_images: torch.Tensor,
    original_images: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    latent_dim: int = 256,
) -> ConditionalVAE:
    """Train a conditional VAE to reconstruct originals from obfuscated images."""
    device = obfuscated_images.device
    model = ConditionalVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(obfuscated_images, original_images)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for obf_batch, orig_batch in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(obf_batch)
            loss = vae_loss(recon, orig_batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def vae_reconstruct(
    model: ConditionalVAE, obfuscated_images: torch.Tensor
) -> torch.Tensor:
    """Use trained VAE to attempt reconstruction."""
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(obfuscated_images)
    return recon
