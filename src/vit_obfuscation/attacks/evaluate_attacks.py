from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from .lbfgs_inversion import lbfgs_inversion_attack
from .mi_fgsm import SimpleUNet, mi_fgsm_attack, train_reconstruction_model
from .vae_reconstruction import train_vae_attack, vae_reconstruct

logger = logging.getLogger(__name__)


def compute_psnr(
    img1: torch.Tensor, img2: torch.Tensor, data_range: float = 2.0
) -> float:
    """Compute PSNR between two image tensors. Assumes range [-1, 1] so data_range=2."""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(data_range**2 / mse)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute mean SSIM between two batches of images."""
    # Simple SSIM implementation
    C1 = (0.01 * 2) ** 2  # data_range = 2
    C2 = (0.03 * 2) ** 2

    mu1 = F.avg_pool2d(img1, 11, 1, 5)
    mu2 = F.avg_pool2d(img2, 11, 1, 5)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1**2, 11, 1, 5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2**2, 11, 1, 5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 11, 1, 5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean().item()


@dataclass
class AttackResult:
    attack_name: str
    ssim: float
    psnr: float
    reconstructed: torch.Tensor


def evaluate_all_attacks(
    obfuscator: ChannelWisePatchLevelObfuscator,
    original_images: torch.Tensor,
    obfuscated_images: torch.Tensor,
    unet_epochs: int = 50,
    vae_epochs: int = 50,
    mi_fgsm_steps: int = 100,
    lbfgs_iterations: int = 200,
    lbfgs_restarts: int = 3,
) -> list[AttackResult]:
    """Run all attacks and return SSIM/PSNR results."""
    results = []

    # Baseline: obfuscated vs original
    ssim_base = compute_ssim(obfuscated_images, original_images)
    psnr_base = compute_psnr(obfuscated_images, original_images)
    results.append(
        AttackResult("obfuscated (no attack)", ssim_base, psnr_base, obfuscated_images)
    )
    logger.info(f"Baseline — SSIM: {ssim_base:.4f}, PSNR: {psnr_base:.2f} dB")

    # 1. MI-FGSM
    logger.info("Running MI-FGSM attack...")
    unet = train_reconstruction_model(
        obfuscated_images, original_images, epochs=unet_epochs
    )
    mi_recon = mi_fgsm_attack(unet, obfuscated_images, iterations=mi_fgsm_steps)
    ssim_mi = compute_ssim(mi_recon, original_images)
    psnr_mi = compute_psnr(mi_recon, original_images)
    results.append(AttackResult("MI-FGSM", ssim_mi, psnr_mi, mi_recon))
    logger.info(f"MI-FGSM — SSIM: {ssim_mi:.4f}, PSNR: {psnr_mi:.2f} dB")

    # 2. L-BFGS
    logger.info("Running L-BFGS inversion attack...")
    lbfgs_recon = lbfgs_inversion_attack(
        obfuscator,
        obfuscated_images,
        max_iterations=lbfgs_iterations,
        num_restarts=lbfgs_restarts,
    )
    ssim_lb = compute_ssim(lbfgs_recon, original_images)
    psnr_lb = compute_psnr(lbfgs_recon, original_images)
    results.append(AttackResult("L-BFGS", ssim_lb, psnr_lb, lbfgs_recon))
    logger.info(f"L-BFGS — SSIM: {ssim_lb:.4f}, PSNR: {psnr_lb:.2f} dB")

    # 3. VAE
    logger.info("Running VAE reconstruction attack...")
    vae = train_vae_attack(obfuscated_images, original_images, epochs=vae_epochs)
    vae_recon = vae_reconstruct(vae, obfuscated_images)
    ssim_vae = compute_ssim(vae_recon, original_images)
    psnr_vae = compute_psnr(vae_recon, original_images)
    results.append(AttackResult("Adversarial VAE", ssim_vae, psnr_vae, vae_recon))
    logger.info(f"VAE — SSIM: {ssim_vae:.4f}, PSNR: {psnr_vae:.2f} dB")

    return results
