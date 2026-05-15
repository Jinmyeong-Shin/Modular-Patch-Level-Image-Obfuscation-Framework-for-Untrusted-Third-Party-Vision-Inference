from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _deranged_restricted_permutation(
    size: int,
    fixed_count: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build a permutation with exactly fixed_count fixed locations when possible."""
    if fixed_count < 0 or fixed_count > size:
        raise ValueError(f"fixed_count must be in [0, {size}], got {fixed_count}")

    perm = torch.arange(size)
    if fixed_count == size:
        return perm

    fixed_locations = set(
        torch.randperm(size, generator=generator)[:fixed_count].tolist()
    )
    movable = torch.tensor(
        [index for index in range(size) if index not in fixed_locations],
        dtype=torch.long,
    )

    if movable.numel() <= 1:
        return perm

    shuffled = movable[torch.randperm(movable.numel(), generator=generator)]
    # Avoid accidental extra fixed points so the security/utility setting is explicit.
    for _ in range(16):
        if not torch.any(shuffled == movable):
            break
        shuffled = movable[torch.randperm(movable.numel(), generator=generator)]
    if torch.any(shuffled == movable):
        shuffled = torch.roll(movable, shifts=1)

    perm[movable] = shuffled
    return perm


def _inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device)
    return inverse


@dataclass(frozen=True)
class HorioPermutationConfig:
    """Restricted random permutation setting from Horio et al. 2024."""

    patch_size: int = 16
    fixed_blocks: int = 120
    fixed_pixels: int = 500
    seed: int = 42


class HorioRestrictedPermutationObfuscator(nn.Module):
    """Restricted block/pixel permutation baseline for ViT patch images.

    Horio et al. encrypt each image by permuting ViT-size blocks and permuting
    pixel-channel positions inside each block. This module models that operation
    on tensors in any numeric range; the transform is lossless and invertible
    when the two secret permutations are known.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int] = 224,
        num_channels: int = 3,
        config: HorioPermutationConfig | None = None,
    ) -> None:
        super().__init__()
        if isinstance(image_size, int):
            height = width = image_size
        else:
            height, width = image_size

        self.height = int(height)
        self.width = int(width)
        self.num_channels = int(num_channels)
        self.config = config or HorioPermutationConfig()
        self.patch_size = int(self.config.patch_size)

        if self.height % self.patch_size != 0 or self.width % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size for Horio baseline"
            )

        num_blocks = (self.height // self.patch_size) * (
            self.width // self.patch_size
        )
        block_width = self.num_channels * self.patch_size * self.patch_size
        generator = torch.Generator().manual_seed(int(self.config.seed))

        block_perm = _deranged_restricted_permutation(
            num_blocks,
            int(self.config.fixed_blocks),
            generator=generator,
        )
        pixel_perm = _deranged_restricted_permutation(
            block_width,
            int(self.config.fixed_pixels),
            generator=generator,
        )

        self.register_buffer("block_permutation", block_perm)
        self.register_buffer("pixel_permutation", pixel_perm)
        self.register_buffer(
            "inverse_block_permutation",
            _inverse_permutation(block_perm),
        )
        self.register_buffer(
            "inverse_pixel_permutation",
            _inverse_permutation(pixel_perm),
        )

    @property
    def num_blocks(self) -> int:
        return int(self.block_permutation.numel())

    @property
    def block_width(self) -> int:
        return int(self.pixel_permutation.numel())

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        if channels != self.num_channels or height != self.height or width != self.width:
            raise ValueError(
                "input tensor shape does not match Horio baseline configuration"
            )
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size
        return (
            images.reshape(
                batch,
                channels,
                patches_h,
                self.patch_size,
                patches_w,
                self.patch_size,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, patches_h * patches_w, self.block_width)
        )

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        patches_h = self.height // self.patch_size
        patches_w = self.width // self.patch_size
        return (
            patches.reshape(
                batch,
                patches_h,
                patches_w,
                self.num_channels,
                self.patch_size,
                self.patch_size,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, self.num_channels, self.height, self.width)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(images)
        encrypted = patches[:, self.block_permutation]
        encrypted = encrypted[:, :, self.pixel_permutation]
        return self._unpatchify(encrypted)

    def inverse(self, encrypted_images: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(encrypted_images)
        recovered = patches[:, :, self.inverse_pixel_permutation]
        recovered = recovered[:, self.inverse_block_permutation]
        return self._unpatchify(recovered)

    def fixed_fraction_summary(self) -> dict[str, float | int]:
        block_indices = torch.arange(
            self.num_blocks,
            device=self.block_permutation.device,
        )
        pixel_indices = torch.arange(
            self.block_width,
            device=self.pixel_permutation.device,
        )
        block_fixed = int(
            torch.sum(self.block_permutation == block_indices).item()
        )
        pixel_fixed = int(
            torch.sum(self.pixel_permutation == pixel_indices).item()
        )
        return {
            "num_blocks": self.num_blocks,
            "block_width": self.block_width,
            "fixed_blocks": block_fixed,
            "fixed_pixels": pixel_fixed,
            "fixed_block_fraction": block_fixed / self.num_blocks,
            "fixed_pixel_fraction": pixel_fixed / self.block_width,
        }
