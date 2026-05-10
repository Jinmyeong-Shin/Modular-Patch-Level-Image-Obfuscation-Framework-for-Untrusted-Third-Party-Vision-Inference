from types import SimpleNamespace

import torch
import torch.nn as nn


class ObfuscationPatchEmbedding(nn.Module):
    """
    Processes obfuscated image patches into embeddings via per-position
    linear decoding and channel merging.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int],
        num_channels: int,
        patch_size: int,
        embed_dim: int,
    ) -> None:
        super().__init__()

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            image_size = tuple(image_size)

        assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0, (
            "Input size must be divisible by patch size"
        )

        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        H, W = self.image_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        patch_dim = patch_size * patch_size

        self.decode_weights = nn.Parameter(
            torch.empty(self.num_patches, embed_dim, patch_dim)
        )
        self.decode_biases = nn.Parameter(torch.empty(self.num_patches, embed_dim))

        self.merge_weights = nn.Parameter(torch.empty(self.num_patches, num_channels))
        self.merge_biases = nn.Parameter(torch.empty(self.num_patches))

        nn.init.normal_(self.decode_weights, std=0.01)
        nn.init.normal_(self.merge_weights, std=0.01)
        nn.init.zeros_(self.decode_biases)
        nn.init.zeros_(self.merge_biases)

        # Dummy projection for HuggingFace ViT compatibility (dtype check)
        self.projection = SimpleNamespace(weight=torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {C}.")

        ps = self.patch_size
        Nh, Nw = H // ps, W // ps
        num_patches = Nh * Nw

        # Patch extraction: (B, C, H, W) -> (B, C, num_patches, ps*ps)
        x_patched = (
            x.view(B, C, Nh, ps, Nw, ps)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(B, C, num_patches, ps * ps)
        )

        # Per-position patch decoding
        decoded_patches = torch.einsum(
            "bcnp,nfp->bcnf", x_patched, self.decode_weights
        ) + self.decode_biases.view(1, 1, num_patches, -1)

        # Channel merging
        x_permuted = decoded_patches.permute(0, 2, 3, 1)
        return torch.einsum(
            "bnec,nc->bne", x_permuted, self.merge_weights
        ) + self.merge_biases.view(1, num_patches, -1)
