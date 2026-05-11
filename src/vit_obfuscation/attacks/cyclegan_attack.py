from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Single residual block with instance normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """ResNet-based generator with 6 residual blocks and instance normalization.

    Architecture: initial conv (7x7) -> 2 downsampling convs -> 6 residual
    blocks -> 2 upsampling deconvs -> final conv (7x7) with Tanh.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()

        # Initial convolution block
        initial = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        downsampling = [
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        # Residual blocks
        residual = [ResidualBlock(256) for _ in range(6)]

        # Upsampling
        upsampling = [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Final convolution
        final = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(
            *initial, *downsampling, *residual, *upsampling, *final
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator.

    Outputs a patch of real/fake predictions rather than a single scalar.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_cyclegan_attack(
    obfuscated_images: torch.Tensor,
    original_images: torch.Tensor,
    epochs: int = 100,
    lr: float = 2e-4,
) -> CycleGANGenerator:
    """Train a CycleGAN to reconstruct originals from obfuscated images.

    Uses adversarial loss (MSE) + cycle-consistency loss (L1, weight=10)
    + identity loss (L1, weight=5).
    """
    device = obfuscated_images.device

    # Generators: G maps obfuscated -> original, F maps original -> obfuscated
    gen_g = CycleGANGenerator().to(device)
    gen_f = CycleGANGenerator().to(device)
    disc_orig = PatchGANDiscriminator().to(device)
    disc_obf = PatchGANDiscriminator().to(device)

    optimizer_g = torch.optim.Adam(
        list(gen_g.parameters()) + list(gen_f.parameters()),
        lr=lr,
        betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        list(disc_orig.parameters()) + list(disc_obf.parameters()),
        lr=lr,
        betas=(0.5, 0.999),
    )

    adversarial_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    dataset = torch.utils.data.TensorDataset(obfuscated_images, original_images)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    lambda_cycle = 10.0
    lambda_identity = 5.0

    gen_g.train()
    gen_f.train()
    disc_orig.train()
    disc_obf.train()

    for epoch in range(epochs):
        for obf_batch, orig_batch in loader:
            # ---------------------
            # Train generators
            # ---------------------
            optimizer_g.zero_grad()

            # Identity loss
            id_orig = gen_g(orig_batch)
            loss_id_orig = identity_loss(id_orig, orig_batch) * lambda_identity
            id_obf = gen_f(obf_batch)
            loss_id_obf = identity_loss(id_obf, obf_batch) * lambda_identity

            # GAN loss
            fake_orig = gen_g(obf_batch)
            pred_fake_orig = disc_orig(fake_orig)
            target_real = torch.ones_like(pred_fake_orig)
            loss_gan_g = adversarial_loss(pred_fake_orig, target_real)

            fake_obf = gen_f(orig_batch)
            pred_fake_obf = disc_obf(fake_obf)
            target_real = torch.ones_like(pred_fake_obf)
            loss_gan_f = adversarial_loss(pred_fake_obf, target_real)

            # Cycle consistency loss
            recovered_obf = gen_f(fake_orig)
            loss_cycle_obf = cycle_loss(recovered_obf, obf_batch) * lambda_cycle
            recovered_orig = gen_g(fake_obf)
            loss_cycle_orig = cycle_loss(recovered_orig, orig_batch) * lambda_cycle

            loss_g = (
                loss_gan_g
                + loss_gan_f
                + loss_cycle_obf
                + loss_cycle_orig
                + loss_id_orig
                + loss_id_obf
            )
            loss_g.backward()
            optimizer_g.step()

            # ---------------------
            # Train discriminators
            # ---------------------
            optimizer_d.zero_grad()

            # Discriminator for original domain
            pred_real_orig = disc_orig(orig_batch)
            target_real = torch.ones_like(pred_real_orig)
            loss_d_real_orig = adversarial_loss(pred_real_orig, target_real)

            pred_fake_orig = disc_orig(fake_orig.detach())
            target_fake = torch.zeros_like(pred_fake_orig)
            loss_d_fake_orig = adversarial_loss(pred_fake_orig, target_fake)

            # Discriminator for obfuscated domain
            pred_real_obf = disc_obf(obf_batch)
            target_real = torch.ones_like(pred_real_obf)
            loss_d_real_obf = adversarial_loss(pred_real_obf, target_real)

            pred_fake_obf = disc_obf(fake_obf.detach())
            target_fake = torch.zeros_like(pred_fake_obf)
            loss_d_fake_obf = adversarial_loss(pred_fake_obf, target_fake)

            loss_d = (
                loss_d_real_orig + loss_d_fake_orig + loss_d_real_obf + loss_d_fake_obf
            ) * 0.5
            loss_d.backward()
            optimizer_d.step()

    return gen_g


def cyclegan_reconstruct(
    generator: CycleGANGenerator, obfuscated_images: torch.Tensor
) -> torch.Tensor:
    """Use trained CycleGAN generator to attempt reconstruction."""
    generator.eval()
    with torch.no_grad():
        reconstructed = generator(obfuscated_images)
    return reconstructed
