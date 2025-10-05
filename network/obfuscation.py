import torch
import torch.nn as nn
import torch.nn.functional as F

from data.obfusccation_output import ObfuscationOutput

class ChannelWisePatchLevelObfuscator(nn.Module):
    """
    An image obfuscation layer that applies a series of fixed, but random, transformations
    to an input image at the patch and channel level. This module is designed to be
    non-trainable, with its transformation parameters (permutations, linear kernels)
    generated once at initialization and stored as buffers.

    The obfuscation process involves two main vectorized steps:
    1.  **Patch-wise Linear Transformation**: Each image patch is transformed by a unique
        linear projection. The specific projection used for a patch is determined by its
        spatial location, cycling through a pre-defined set of `group_size`
        transformation kernels. A different set of kernels is used for each channel.
        This implementation is equivalent with Conv2D layers in original paper but optimized.
    2.  **Channel Permutation**: The order of the final image channels is shuffled.

    The entire process is implemented using efficient, vectorized tensor operations
    (e.g., `torch.einsum`, `torch.gather`) to ensure high performance. The final output
    is passed through a `tanh` activation to bound the values between -1 and 1.
    """
    def __init__(
        self,
        image_size: int | tuple[int, int],
        num_channels: int,
        patch_size: int,
        group_size: int,
    ) -> None:
        super().__init__()

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            image_size = tuple(image_size)

        assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0, \
            'Input size must be divisible by patch size'

        self.image_size = image_size
        self.num_channels = num_channels 
        self.patch_size = patch_size
        self.group_size = group_size

        H, W = self.image_size
        num_patches = (H // patch_size) * (W // patch_size)

        # Create obfuscation kernels. A unique set of group_size kernels is created for each channel.
        # These are effectively linear layers on flattened patches. We create them temporarily
        # to get randomly initialized weights.
        obfuscation_kernels = [
            nn.Conv2d(
                in_channels=1,
                out_channels=patch_size * patch_size,
                kernel_size=patch_size,
                stride=patch_size
            ) for _ in range(num_channels * group_size)
        ]

        # Group weights into a single tensor for cleaner state management.
        weights = []
        biases = []
        for kernel in obfuscation_kernels:
            # The conv kernel is (ps*ps, 1, ps, ps). We view it as a linear layer weight
            # of shape (ps*ps, ps*ps) for the transformation on a flattened patch.
            weights.append(kernel.weight.view(patch_size * patch_size, patch_size * patch_size).detach())
            biases.append(kernel.bias.detach())

        # register_buffer ensures these tensors are part of the module's state
        # but are not considered parameters to be trained.
        # Shape: (num_channels, group_size, ps*ps, ps*ps)
        obfuscation_weights = torch.stack(weights).view(
            num_channels, group_size, patch_size * patch_size, patch_size * patch_size
        )
        obfuscation_biases = torch.stack(biases).view(num_channels, group_size, patch_size * patch_size)

        # nn.init.uniform_(obfuscation_weights, a=-0.05, b=0.05)
        # nn.init.zeros_(obfuscation_biases)
        # nn.init.normal_(obfuscation_biases, std=0.2)

        self.register_buffer('obfuscation_weights', obfuscation_weights)
        self.register_buffer('obfuscation_biases', obfuscation_biases)

        # Create and stack inter-patch permutations for each channel.
        patch_permutations = [torch.randperm(num_patches) for _ in range(num_channels)]
        self.register_buffer('patch_permutations', torch.stack(patch_permutations))

        # Create and register the channel permutation.
        channel_permutation = torch.randperm(num_channels)
        self.register_buffer('channel_permutation', channel_permutation)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the obfuscation transformations to an input batch of images.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Obfuscated tensor of shape (B, C, H, W) with values in [-1, 1].
        """
        B, C, H, W = x.shape

        assert C == self.num_channels, f'Number of input channels ({C}) does not match num_channels ({self.num_channels})'
        assert (H, W) == self.image_size, f'Input H/W ({H}, {W}) does not match expected image_size {self.image_size}'

        ps = self.patch_size
        Nh, Nw = H // ps, W // ps
        num_patches = Nh * Nw

        # 1. Vectorized Patch Extraction
        # Reshape the input image into a sequence of flattened patches.
        # (B, C, H, W) -> (B, C, num_patches, ps*ps)
        x_patched = x.view(B, C, Nh, ps, Nw, ps).permute(0, 1, 2, 4, 3, 5).reshape(B, C, num_patches, ps * ps)

        # 2. Vectorized Patch-wise Linear Transformation
        # Each patch is transformed by a linear layer. The specific layer used is
        # determined by the patch's spatial location (cycling through `group_size` kernels)
        # and its channel.

        # Create a map to select a kernel for each patch based on its grid position.
        # This creates a repeating diagonal pattern of kernel indices.
        ridx = torch.arange(Nh, device=x.device).view(Nh, 1)
        cidx = torch.arange(Nw, device=x.device).view(1, Nw)
        group_map = ((ridx + cidx) % self.group_size).view(-1)  # Shape: (num_patches)

        # Gather the weights for each channel and patch location using the map.
        # Shape: (C, num_patches, ps*ps, ps*ps)
        selected_weights = self.obfuscation_weights[:, group_map]
        selected_biases = self.obfuscation_biases[:, group_map]

        # Apply the linear transformation using einsum for batched matrix multiplication. For each
        # item in the batch, this applies a unique linear layer to each patch of each channel:
        # patch_vector[c, n] @ W[c, n].
        # 'bcnp,cnpo->bcno' computes this efficiently.
        obfuscated_patches = torch.einsum('bcnp,cnpo->bcno', x_patched, selected_weights) + selected_biases

        # 3. Vectorized Un-patching (reshaping back to image format)
        # The sequence of permuted patches is reshaped back into an image grid. This
        # is the inverse operation of the patch extraction in step 1.
        x_out = obfuscated_patches.view(B, C, Nh, Nw, ps, ps)
        x_out = x_out.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)

        # 4. Channel Permutation
        # The channels of the final image are shuffled using the pre-defined permutation.
        x_out = x_out[:, self.channel_permutation]
        
        # 5. Final Activation
        # A tanh activation is applied to bound the output values to [-1, 1].
        return F.tanh(x_out)
