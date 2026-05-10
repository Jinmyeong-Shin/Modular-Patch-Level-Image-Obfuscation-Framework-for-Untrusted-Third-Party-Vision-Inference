from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator


def lbfgs_inversion_attack(
    obfuscator: ChannelWisePatchLevelObfuscator,
    obfuscated_images: torch.Tensor,
    max_iterations: int = 500,
    num_restarts: int = 3,
    lr: float = 0.1,
) -> torch.Tensor:
    """
    L-BFGS optimization attack: given obfuscated image Y, try to find X
    such that Obfuscate(X) ≈ Y.

    Uses multiple random restarts and returns the best reconstruction.
    Should fail due to non-invertibility (tanh + permutations).
    """
    device = obfuscated_images.device
    B, C, H, W = obfuscated_images.shape
    best_recon = torch.randn_like(obfuscated_images)
    best_loss = float("inf")

    for restart in range(num_restarts):
        # Random initialization
        x = torch.randn(B, C, H, W, device=device, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [x],
            lr=lr,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )

        for step in range(max_iterations // 20):

            def closure():
                optimizer.zero_grad()
                # Temporarily enable gradients through obfuscator
                with torch.enable_grad():
                    y_hat = _obfuscate_with_grad(obfuscator, x)
                    loss = F.mse_loss(y_hat, obfuscated_images)
                loss.backward()
                return loss

            loss = optimizer.step(closure)

        final_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        if final_loss < best_loss:
            best_loss = final_loss
            best_recon = x.detach().clone()

    return best_recon.clamp(-1, 1)


def _obfuscate_with_grad(
    obfuscator: ChannelWisePatchLevelObfuscator,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run obfuscation with gradients enabled (bypass @torch.no_grad)."""
    B, C, H, W = x.shape
    ps = obfuscator.patch_size
    Nh, Nw = H // ps, W // ps

    x_patched = (
        x.view(B, C, Nh, ps, Nw, ps)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(B, C, Nh * Nw, ps * ps)
    )

    ridx = torch.arange(Nh, device=x.device).view(Nh, 1)
    cidx = torch.arange(Nw, device=x.device).view(1, Nw)
    group_map = ((ridx + cidx) % obfuscator.group_size).view(-1)

    selected_weights = obfuscator.obfuscation_weights[:, group_map]
    selected_biases = obfuscator.obfuscation_biases[:, group_map]

    obfuscated_patches = (
        torch.einsum("bcnp,cnpo->bcno", x_patched, selected_weights) + selected_biases
    )

    x_out = obfuscated_patches.view(B, C, Nh, Nw, ps, ps)
    x_out = x_out.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    x_out = x_out[:, obfuscator.channel_permutation]
    return torch.tanh(x_out)
