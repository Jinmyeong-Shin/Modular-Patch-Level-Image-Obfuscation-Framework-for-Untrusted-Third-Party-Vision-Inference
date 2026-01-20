import torch
import torch.nn as NN
import torch.nn.functional as F

from safetensors.torch import load_file

from ..data_model.config import ObfuscatorConfig

class Obfuscator(NN.Module):
    """
    An image obfuscation layer that applies a series of fixed, but random, transformations
    to an input image at the patch and channel level. This module is designed to be
    non-trainable, with its transformation parameters (permutations, linear kernels)
    generated once at initialization and stored as buffers. The configuration can be
    provided via a dataclass, a dictionary, or a YAML file.
    """
    def __init__(self, config: ObfuscatorConfig) -> None:
        """
        Initialize the Obfuscator layer from a configuration object.

        Args:
            config (ObfuscatorConfig): The configuration object for the obfuscator.
        """
        super().__init__()
        self.config = config

        image_size = config.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        patch_size = config.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        # Validate the input image can be divided by patch size
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, \
            f'Image size {image_size} must be divisible by patch size {patch_size}'
        
        self.image_size = image_size
        self.patch_size = patch_size

        h, w = self.image_size
        ph, pw = self.patch_size
        self.num_patches = (h // ph) * (w // pw)
        self.num_pixels_per_patch = ph * pw

        if self.config.kernels_path:
            self._load_kernels(self.config.kernels_path)
        else:
            self._init_kernels()

    def _init_kernels(self) -> None:
        """
        Initializes the obfuscation kernels and channel permutation.
        These parameters are generated randomly and registered as buffers,
        making them part of the module's state but not trainable.
        """
        # Shape: (C, num_kernels, patch_pixels, patch_pixels)
        weights_shape = (
            self.config.num_channels,
            self.config.num_kernels,
            self.num_pixels_per_patch,
            self.num_pixels_per_patch
        )
        # Shape: (C, num_kernels, patch_pixels)
        biases_shape = (
            self.config.num_channels,
            self.config.num_kernels,
            self.num_pixels_per_patch
        )

        kernel_weights = torch.randn(weights_shape)
        kernel_biases = torch.randn(biases_shape)

        self.register_buffer('kernel_weights', kernel_weights)
        self.register_buffer('kernel_biases', kernel_biases)

        channel_perm = torch.randperm(self.config.num_channels)
        self.register_buffer('channel_perm', channel_perm)

    def _load_kernels(self, path: str) -> None:
        """
        Loads kernel weights and biases from a .safetensors file and registers them as buffers.
        
        Args:
            path (str): The path to the .safetensors file.
        """
        state_dict = load_file(path)

        expected_weights_shape = (
            self.config.num_channels,
            self.config.num_kernels,
            self.num_pixels_per_patch,
            self.num_pixels_per_patch
        )
        assert state_dict['kernel_weights'].shape == expected_weights_shape, \
            f"Loaded kernel_weights shape {state_dict['kernel_weights'].shape} does not match expected shape {expected_weights_shape}"

        expected_biases_shape = (
            self.config.num_channels,
            self.config.num_kernels,
            self.num_pixels_per_patch
        )
        assert state_dict['kernel_biases'].shape == expected_biases_shape, \
            f"Loaded kernel_biases shape {state_dict['kernel_biases'].shape} does not match expected shape {expected_biases_shape}"

        expected_channel_perm_shape = (self.config.num_channels,)
        assert state_dict['channel_perm'].shape == expected_channel_perm_shape, \
            f"Loaded channel_perm shape {state_dict['channel_perm'].shape} does not match expected shape {expected_channel_perm_shape}"

        self.register_buffer('kernel_weights', state_dict['kernel_weights'])
        self.register_buffer('kernel_biases', state_dict['kernel_biases'])
        self.register_buffer('channel_perm', state_dict['channel_perm'])

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
        
        # Input Validation
        assert C == self.config.num_channels, \
            f'Number of input channels ({C}) does not match num_channels ({self.config.num_channels})'
        assert (H, W) == self.image_size, \
            f'Input H/W ({H}, {W}) does not match expected image_size {self.image_size}'

        ph, pw = self.patch_size
        
        Nh, Nw = H // ph, W // pw

        # 1. Vectorized Patch Extraction
        # Reshape the input image into a sequence of flattened patches.
        # (B, C, H, W) -> (B, C, num_patches, ph*pw)
        x_patched = x.view(B, C, Nh, ph, Nw, pw).permute(0, 1, 2, 4, 3, 5).reshape(B, C, self.num_patches, self.num_pixels_per_patch)

        # 2. Vectorized Patch-wise Linear Transformation
        # Create a map to select a kernel for each patch based on its grid position.
        # This creates a repeating diagonal pattern of kernel indices.
        ridx = torch.arange(Nh, device=x.device).view(Nh, 1)
        cidx = torch.arange(Nw, device=x.device).view(1, Nw)
        group_map = ((ridx + cidx) % self.config.num_kernels).view(-1)  # Shape: (num_patches)

        # Gather the weights and biases for each channel and patch location.
        selected_weights = self.kernel_weights[:, group_map]
        selected_biases = self.kernel_biases[:, group_map]

        # Apply the linear transformation using einsum for batched matrix multiplication.
        # 'bcnp,cnpo->bcno' means for each item in batch (b) and channel (c),
        # we multiply a patch vector (p) by its corresponding kernel matrix (o).
        obfuscated_patches = torch.einsum('bcnp,cnpo->bcno', x_patched, selected_weights) + selected_biases

        # 3. Vectorized Un-patching (reshaping back to image format)
        x_out = obfuscated_patches.view(B, C, Nh, Nw, ph, pw)
        x_out = x_out.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)

        # 4. Channel Permutation
        # The channels of the final image are shuffled.
        x_out = x_out[:, self.channel_perm]
        
        # 5. Final Activation
        return F.tanh(x_out)
        
