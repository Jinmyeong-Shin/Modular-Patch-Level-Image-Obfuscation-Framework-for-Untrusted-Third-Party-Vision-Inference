import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWisePatchLevelObfuscator(nn.Module):
    """
    Non-trainable image obfuscation layer that applies fixed random transformations
    at the patch and channel level.

    Steps:
    1. Patch-wise linear transformation (unique per spatial position, cycling through group_size kernels)
    2. Optional per-channel patch permutation
    3. Channel permutation
    4. Tanh activation to bound output to [-1, 1]
    """

    def __init__(
        self,
        image_size: int | tuple[int, int],
        num_channels: int,
        patch_size: int,
        group_size: int,
        use_tanh: bool = True,
        apply_patch_permutation: bool = False,
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
        self.group_size = group_size
        self.use_tanh = use_tanh
        self.apply_patch_permutation = apply_patch_permutation

        H, W = self.image_size
        num_patches = (H // patch_size) * (W // patch_size)

        obfuscation_kernels = [
            nn.Conv2d(
                in_channels=1,
                out_channels=patch_size * patch_size,
                kernel_size=patch_size,
                stride=patch_size,
            )
            for _ in range(num_channels * group_size)
        ]

        weights = []
        biases = []
        for kernel in obfuscation_kernels:
            weights.append(
                kernel.weight.view(
                    patch_size * patch_size, patch_size * patch_size
                ).detach()
            )
            biases.append(kernel.bias.detach())

        obfuscation_weights = torch.stack(weights).view(
            num_channels, group_size, patch_size * patch_size, patch_size * patch_size
        )
        obfuscation_biases = torch.stack(biases).view(
            num_channels, group_size, patch_size * patch_size
        )

        self.register_buffer("obfuscation_weights", obfuscation_weights)
        self.register_buffer("obfuscation_biases", obfuscation_biases)

        patch_permutations = [torch.randperm(num_patches) for _ in range(num_channels)]
        self.register_buffer("patch_permutations", torch.stack(patch_permutations))

        channel_permutation = torch.randperm(num_channels)
        self.register_buffer("channel_permutation", channel_permutation)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        assert C == self.num_channels, (
            f"Number of input channels ({C}) does not match num_channels ({self.num_channels})"
        )
        assert (H, W) == self.image_size, (
            f"Input H/W ({H}, {W}) does not match expected image_size {self.image_size}"
        )

        ps = self.patch_size
        Nh, Nw = H // ps, W // ps

        # Patch extraction: (B, C, H, W) -> (B, C, num_patches, ps*ps)
        x_patched = (
            x.view(B, C, Nh, ps, Nw, ps)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(B, C, Nh * Nw, ps * ps)
        )

        # Kernel selection based on spatial position
        ridx = torch.arange(Nh, device=x.device).view(Nh, 1)
        cidx = torch.arange(Nw, device=x.device).view(1, Nw)
        group_map = ((ridx + cidx) % self.group_size).view(-1)

        selected_weights = self.obfuscation_weights[:, group_map]
        selected_biases = self.obfuscation_biases[:, group_map]

        # Patch-wise linear transformation
        obfuscated_patches = (
            torch.einsum("bcnp,cnpo->bcno", x_patched, selected_weights)
            + selected_biases
        )

        # The original PoC generated patch permutations but did not apply them.
        # Keep that behavior by default because the learned deobfuscation embedding
        # assumes local spatial correspondence between input and ViT patch grids.
        if self.apply_patch_permutation:
            permuted_patches = torch.empty_like(obfuscated_patches)
            for channel_idx in range(C):
                permuted_patches[:, channel_idx] = obfuscated_patches[
                    :, channel_idx, self.patch_permutations[channel_idx]
                ]
        else:
            permuted_patches = obfuscated_patches

        # Reshape back to image
        x_out = permuted_patches.view(B, C, Nh, Nw, ps, ps)
        x_out = x_out.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)

        # Channel permutation
        x_out = x_out[:, self.channel_permutation]

        if self.use_tanh:
            return F.tanh(x_out)
        return x_out
