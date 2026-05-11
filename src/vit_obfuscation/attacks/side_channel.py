from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class SideChannelResult:
    """Combined results from all side-channel leakage analyses."""

    frequency_correlation: float
    spatial_autocorrelation: float
    histogram_kl_divergence: float
    mutual_information: float
    original_spectrum: torch.Tensor
    obfuscated_spectrum: torch.Tensor


def compute_frequency_leakage(
    original: torch.Tensor,
    obfuscated: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Compute 2D FFT of both inputs and measure Pearson correlation between
    log magnitude spectra, averaged over batch and channels.

    Returns:
        (correlation, original_spectrum, obfuscated_spectrum)
    """
    original = original.detach().cpu().float()
    obfuscated = obfuscated.detach().cpu().float()

    orig_fft = torch.fft.fft2(original)
    obf_fft = torch.fft.fft2(obfuscated)

    orig_spectrum = torch.fft.fftshift(orig_fft.abs())
    obf_spectrum = torch.fft.fftshift(obf_fft.abs())

    # Log magnitude (add epsilon for numerical stability)
    orig_log = torch.log(orig_spectrum + 1e-10)
    obf_log = torch.log(obf_spectrum + 1e-10)

    # Pearson correlation averaged over batch and channels
    correlations = []
    for b in range(original.shape[0]):
        for c in range(original.shape[1]):
            x = orig_log[b, c].flatten()
            y = obf_log[b, c].flatten()
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            num = (x_centered * y_centered).sum()
            denom = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
            if denom < 1e-12:
                correlations.append(0.0)
            else:
                correlations.append((num / denom).item())

    correlation = sum(correlations) / len(correlations)
    return correlation, orig_spectrum, obf_spectrum


def compute_spatial_autocorrelation(
    original: torch.Tensor,
    obfuscated: torch.Tensor,
) -> float:
    """Compute patch-wise mean vectors for both original and obfuscated, then
    measure Pearson correlation of spatial patterns.

    This checks if the obfuscated image retains spatial structure from the
    original.
    """
    original = original.detach().cpu().float()
    obfuscated = obfuscated.detach().cpu().float()

    # Use non-overlapping patches of size 16x16 (or smaller if image is small)
    patch_size = min(16, original.shape[2], original.shape[3])

    def _patch_means(tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = tensor.shape
        # Trim to multiple of patch_size
        h_trimmed = (H // patch_size) * patch_size
        w_trimmed = (W // patch_size) * patch_size
        trimmed = tensor[:, :, :h_trimmed, :w_trimmed]
        # Reshape into patches and compute mean per patch
        patches = trimmed.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        # patches shape: (B, C, n_h, n_w, patch_size, patch_size)
        return patches.mean(dim=(-2, -1))  # (B, C, n_h, n_w)

    orig_means = _patch_means(original)
    obf_means = _patch_means(obfuscated)

    # Pearson correlation averaged over batch and channels
    correlations = []
    for b in range(original.shape[0]):
        for c in range(original.shape[1]):
            x = orig_means[b, c].flatten()
            y = obf_means[b, c].flatten()
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            num = (x_centered * y_centered).sum()
            denom = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
            if denom < 1e-12:
                correlations.append(0.0)
            else:
                correlations.append((num / denom).item())

    return sum(correlations) / len(correlations)


def compute_histogram_leakage(
    original: torch.Tensor,
    obfuscated: torch.Tensor,
    num_bins: int = 256,
) -> float:
    """Compute per-channel pixel intensity histograms for both inputs and
    return the mean KL divergence across channels and batch.
    """
    original = original.detach().cpu().float()
    obfuscated = obfuscated.detach().cpu().float()

    eps = 1e-10
    min_val = min(original.min().item(), obfuscated.min().item())
    max_val = max(original.max().item(), obfuscated.max().item())

    kl_divs = []
    for b in range(original.shape[0]):
        for c in range(original.shape[1]):
            orig_hist = torch.histc(
                original[b, c], bins=num_bins, min=min_val, max=max_val
            )
            obf_hist = torch.histc(
                obfuscated[b, c], bins=num_bins, min=min_val, max=max_val
            )

            # Normalize to probability distributions
            orig_prob = orig_hist / orig_hist.sum() + eps
            obf_prob = obf_hist / obf_hist.sum() + eps

            # KL(original || obfuscated)
            kl = (orig_prob * torch.log(orig_prob / obf_prob)).sum()
            kl_divs.append(kl.item())

    return sum(kl_divs) / len(kl_divs)


def compute_mutual_information(
    original: torch.Tensor,
    obfuscated: torch.Tensor,
    num_bins: int = 64,
) -> float:
    """Estimate mutual information between original and obfuscated pixel values
    using 2D histogram binning.
    """
    original = original.detach().cpu().float()
    obfuscated = obfuscated.detach().cpu().float()

    eps = 1e-10

    x = original.flatten()
    y = obfuscated.flatten()

    # Bin edges
    x_min, x_max = x.min().item(), x.max().item()
    y_min, y_max = y.min().item(), y.max().item()

    # Digitize into bin indices
    x_bins = (
        ((x - x_min) / (x_max - x_min + eps) * (num_bins - 1))
        .long()
        .clamp(0, num_bins - 1)
    )
    y_bins = (
        ((y - y_min) / (y_max - y_min + eps) * (num_bins - 1))
        .long()
        .clamp(0, num_bins - 1)
    )

    # Joint histogram
    joint = torch.zeros(num_bins, num_bins)
    indices = x_bins * num_bins + y_bins
    joint = joint.flatten()
    ones = torch.ones(indices.shape[0])
    joint.scatter_add_(0, indices, ones)
    joint = joint.reshape(num_bins, num_bins)

    # Normalize to joint probability
    joint_prob = joint / joint.sum() + eps

    # Marginals
    p_x = joint_prob.sum(dim=1)
    p_y = joint_prob.sum(dim=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
    outer = p_x.unsqueeze(1) * p_y.unsqueeze(0)
    mi = (joint_prob * torch.log(joint_prob / outer)).sum()

    return mi.item()


def side_channel_analysis(
    original_images: torch.Tensor,
    obfuscated_images: torch.Tensor,
) -> SideChannelResult:
    """Run all side-channel leakage analyses and return combined result."""
    logger.info("Running frequency leakage analysis...")
    freq_corr, orig_spectrum, obf_spectrum = compute_frequency_leakage(
        original_images, obfuscated_images
    )

    logger.info("Running spatial autocorrelation analysis...")
    spatial_corr = compute_spatial_autocorrelation(original_images, obfuscated_images)

    logger.info("Running histogram leakage analysis...")
    hist_kl = compute_histogram_leakage(original_images, obfuscated_images)

    logger.info("Running mutual information analysis...")
    mi = compute_mutual_information(original_images, obfuscated_images)

    logger.info(
        "Side-channel analysis complete: freq_corr=%.4f, spatial_corr=%.4f, "
        "hist_kl=%.4f, mi=%.4f",
        freq_corr,
        spatial_corr,
        hist_kl,
        mi,
    )

    return SideChannelResult(
        frequency_correlation=freq_corr,
        spatial_autocorrelation=spatial_corr,
        histogram_kl_divergence=hist_kl,
        mutual_information=mi,
        original_spectrum=orig_spectrum,
        obfuscated_spectrum=obf_spectrum,
    )
