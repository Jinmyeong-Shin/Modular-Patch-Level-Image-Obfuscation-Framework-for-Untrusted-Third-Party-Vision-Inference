from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from .cyclegan_attack import cyclegan_reconstruct, train_cyclegan_attack
from .lbfgs_inversion import lbfgs_inversion_attack
from .mi_fgsm import mi_fgsm_attack
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


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 2.0,
) -> float:
    """Compute mean SSIM between two batches of images."""
    # Simple SSIM implementation
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

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
    mse: float
    reconstructed: torch.Tensor


def evaluate_all_attacks(
    obfuscator: ChannelWisePatchLevelObfuscator,
    original_images: torch.Tensor,
    obfuscated_images: torch.Tensor,
    attack_train_original_images: torch.Tensor | None = None,
    attack_train_obfuscated_images: torch.Tensor | None = None,
    unet_epochs: int = 50,
    vae_epochs: int = 50,
    cyclegan_epochs: int = 100,
    mi_fgsm_steps: int = 100,
    lbfgs_iterations: int = 200,
    lbfgs_restarts: int = 3,
    data_range: float = 2.0,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    run_cyclegan: bool = True,
) -> list[AttackResult]:
    """Run all attacks and return SSIM/PSNR results."""
    results = []
    train_original = (
        attack_train_original_images
        if attack_train_original_images is not None
        else original_images
    )
    train_obfuscated = (
        attack_train_obfuscated_images
        if attack_train_obfuscated_images is not None
        else obfuscated_images
    )

    # Baseline: obfuscated vs original
    ssim_base = compute_ssim(obfuscated_images, original_images, data_range=data_range)
    psnr_base = compute_psnr(obfuscated_images, original_images, data_range=data_range)
    mse_base = F.mse_loss(obfuscated_images, original_images).item()
    results.append(
        AttackResult(
            "obfuscated (no attack)",
            ssim_base,
            psnr_base,
            mse_base,
            obfuscated_images,
        )
    )
    logger.info(f"Baseline — SSIM: {ssim_base:.4f}, PSNR: {psnr_base:.2f} dB")

    # 1. MI-FGSM
    logger.info("Running MI-FGSM attack...")
    mi_recon = mi_fgsm_attack(
        obfuscator,
        obfuscated_images,
        iterations=mi_fgsm_steps,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    ssim_mi = compute_ssim(mi_recon, original_images, data_range=data_range)
    psnr_mi = compute_psnr(mi_recon, original_images, data_range=data_range)
    mse_mi = F.mse_loss(mi_recon, original_images).item()
    results.append(AttackResult("MI-FGSM", ssim_mi, psnr_mi, mse_mi, mi_recon))
    logger.info(f"MI-FGSM — SSIM: {ssim_mi:.4f}, PSNR: {psnr_mi:.2f} dB")

    # 2. L-BFGS
    logger.info("Running L-BFGS inversion attack...")
    lbfgs_recon = lbfgs_inversion_attack(
        obfuscator,
        obfuscated_images,
        max_iterations=lbfgs_iterations,
        num_restarts=lbfgs_restarts,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    ssim_lb = compute_ssim(lbfgs_recon, original_images, data_range=data_range)
    psnr_lb = compute_psnr(lbfgs_recon, original_images, data_range=data_range)
    mse_lb = F.mse_loss(lbfgs_recon, original_images).item()
    results.append(AttackResult("L-BFGS", ssim_lb, psnr_lb, mse_lb, lbfgs_recon))
    logger.info(f"L-BFGS — SSIM: {ssim_lb:.4f}, PSNR: {psnr_lb:.2f} dB")

    # 3. VAE
    logger.info("Running VAE reconstruction attack...")
    vae = train_vae_attack(train_obfuscated, train_original, epochs=vae_epochs)
    vae_recon = vae_reconstruct(vae, obfuscated_images)
    ssim_vae = compute_ssim(vae_recon, original_images, data_range=data_range)
    psnr_vae = compute_psnr(vae_recon, original_images, data_range=data_range)
    mse_vae = F.mse_loss(vae_recon, original_images).item()
    results.append(
        AttackResult("Adversarial VAE", ssim_vae, psnr_vae, mse_vae, vae_recon)
    )
    logger.info(f"VAE — SSIM: {ssim_vae:.4f}, PSNR: {psnr_vae:.2f} dB")

    # 4. CycleGAN
    if not run_cyclegan:
        return results

    logger.info("Running CycleGAN reconstruction attack...")
    cyclegan_gen = train_cyclegan_attack(
        train_obfuscated, train_original, epochs=cyclegan_epochs
    )
    cyclegan_recon = cyclegan_reconstruct(cyclegan_gen, obfuscated_images)
    ssim_cg = compute_ssim(cyclegan_recon, original_images, data_range=data_range)
    psnr_cg = compute_psnr(cyclegan_recon, original_images, data_range=data_range)
    mse_cg = F.mse_loss(cyclegan_recon, original_images).item()
    results.append(AttackResult("CycleGAN", ssim_cg, psnr_cg, mse_cg, cyclegan_recon))
    logger.info(f"CycleGAN — SSIM: {ssim_cg:.4f}, PSNR: {psnr_cg:.2f} dB")

    return results
