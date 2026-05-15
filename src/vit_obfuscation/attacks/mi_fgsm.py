from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from .lbfgs_inversion import _obfuscate_with_grad


class SimpleUNet(nn.Module):
    """Lightweight U-Net for image reconstruction."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


def train_reconstruction_model(
    obfuscated_images: torch.Tensor,
    original_images: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
) -> SimpleUNet:
    """Train a U-Net to reconstruct originals from obfuscated images."""
    model = SimpleUNet().to(obfuscated_images.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(obfuscated_images, original_images)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for obf_batch, orig_batch in loader:
            optimizer.zero_grad()
            recon = model(obf_batch)
            loss = F.mse_loss(recon, orig_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def mi_fgsm_attack(
    obfuscator: ChannelWisePatchLevelObfuscator,
    obfuscated_images: torch.Tensor,
    iterations: int = 100,
    step_size: float = 0.01,
    momentum: float = 1.0,
    num_restarts: int = 3,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    MI-FGSM inversion attack: optimize candidate clean images so that their
    obfuscated outputs match the observed obfuscated images.

    Returns the best candidate image found across random restarts.
    """
    best_recon = torch.empty_like(obfuscated_images).uniform_(clip_min, clip_max)
    best_loss = torch.full(
        (obfuscated_images.shape[0],), float("inf"), device=obfuscated_images.device
    )

    for _ in range(num_restarts):
        candidate = torch.empty_like(obfuscated_images).uniform_(clip_min, clip_max)
        candidate.requires_grad_(True)
        grad_momentum = torch.zeros_like(candidate)

        for _ in range(iterations):
            if candidate.grad is not None:
                candidate.grad.zero_()

            predicted = _obfuscate_with_grad(obfuscator, candidate)
            per_image_loss = F.mse_loss(
                predicted, obfuscated_images, reduction="none"
            ).mean(dim=(1, 2, 3))
            loss = per_image_loss.sum()
            loss.backward()

            with torch.no_grad():
                grad = candidate.grad
                grad_norm = grad / (
                    grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8
                )
                grad_momentum = momentum * grad_momentum + grad_norm
                candidate = candidate - step_size * grad_momentum.sign()
                candidate = (
                    candidate.clamp(clip_min, clip_max).detach().requires_grad_(True)
                )

        with torch.no_grad():
            final = _obfuscate_with_grad(obfuscator, candidate)
            per_image_loss = F.mse_loss(
                final, obfuscated_images, reduction="none"
            ).mean(dim=(1, 2, 3))
            improved = per_image_loss < best_loss
            best_loss[improved] = per_image_loss[improved]
            best_recon[improved] = candidate.detach()[improved]

    return best_recon.detach()
