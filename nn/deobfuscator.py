from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file

from ..data_model.config import DeobfuscatorConfig


class DeobfuscatorPatchEmbedding(nn.Module):
    """
    An embedding layer that processes an obfuscated image patch-wise.

    This layer performs two main vectorized operations:
    1.  **Patch Decoding**: Each patch from the input image (across all channels) is
        transformed into a higher-dimensional embedding. A unique linear transformation
        is learned for each patch position.
    2.  **Channel Merging**: The embeddings from the different input channels
        for each patch are merged into a single embedding via a learned linear combination.
        Again, a unique combination is learned for each patch position.
    """

    def __init__(self, config: DeobfuscatorConfig) -> None:
        super().__init__()
        self.config = config

        image_size = config.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        patch_size = config.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.image_size = image_size
        self.patch_size = patch_size

        h, w = self.image_size
        ph, pw = self.patch_size
        self.num_patches = (h // ph) * (w // pw)
        self.num_pixels_per_patch = ph * pw

        # Parameters for patch decoding: (num_patches) x nn.Linear(num_pixels_per_patch, embed_dim)
        self.decode_weights = nn.Parameter(torch.empty(self.num_patches, config.embed_dim, self.num_pixels_per_patch))
        self.decode_biases = nn.Parameter(torch.empty(self.num_patches, config.embed_dim))

        # Parameters for channel merging: (num_patches) x nn.Linear(num_channels, 1)
        self.merge_weights = nn.Parameter(torch.empty(self.num_patches, config.num_channels))
        self.merge_biases = nn.Parameter(torch.empty(self.num_patches))

        if config.weights_path:
            self._load_weights(config.weights_path)
        else:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initializes the trainable weights of the layer."""
        nn.init.normal_(self.decode_weights, std=0.01)
        nn.init.normal_(self.merge_weights, std=0.01)
        nn.init.zeros_(self.decode_biases)
        nn.init.zeros_(self.merge_biases)

    def _load_weights(self, path: str) -> None:
        """
        Loads Deobfuscator weights from a .safetensors file and registers them as buffers.
        
        Args:
            path (str): The path to the .safetensors file.
        """
        state_dict = load_file(path)

        expected_decode_weights_shape = (self.num_patches, self.config.embed_dim, self.num_pixels_per_patch)
        assert state_dict['decode_weights'].shape == expected_decode_weights_shape, \
            f"Loaded decode_weights shape {state_dict['decode_weights'].shape} does not match expected shape {expected_decode_weights_shape}"

        expected_decode_biases_shape = (self.num_patches, self.config.embed_dim)
        assert state_dict['decode_biases'].shape == expected_decode_biases_shape, \
            f"Loaded decode_biases shape {state_dict['decode_biases'].shape} does not match expected shape {expected_decode_biases_shape}"

        expected_merge_weights_shape = (self.num_patches, self.config.num_channels)
        assert state_dict['merge_weights'].shape == expected_merge_weights_shape, \
            f"Loaded merge_weights shape {state_dict['merge_weights'].shape} does not match expected shape {expected_merge_weights_shape}"

        expected_merge_biases_shape = (self.num_patches,)
        assert state_dict['merge_biases'].shape == expected_merge_biases_shape, \
            f"Loaded merge_biases shape {state_dict['merge_biases'].shape} does not match expected shape {expected_merge_biases_shape}"

        self.decode_weights.data = state_dict['decode_weights']
        self.decode_biases.data = state_dict['decode_biases']
        self.merge_weights.data = state_dict['merge_weights']
        self.merge_biases.data = state_dict['merge_biases']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != self.config.num_channels:
            raise ValueError(
                "The channel dimension of the pixel values does not match the one in the configuration. "
                f"Expected {self.config.num_channels}, got {C}."
            )

        ph, pw = self.patch_size
        Nh, Nw = H // ph, W // pw

        # 1. Vectorized Patch Extraction
        x_patched = x.view(B, C, Nh, ph, Nw, pw).permute(0, 1, 2, 4, 3, 5).reshape(B, C, self.num_patches, ph * pw)

        # 2. Vectorized Patch Decoding
        decoded_patches = torch.einsum('bcnp,nfp->bcnf', x_patched, self.decode_weights) + self.decode_biases.view(1, 1, self.num_patches, -1)
        
        # 3. Vectorized Channel Merging
        x_permuted = decoded_patches.permute(0, 2, 3, 1)
        return torch.einsum('bnec,nc->bne', x_permuted, self.merge_weights) + self.merge_biases.view(1, self.num_patches, -1)


class Deobfuscator(nn.Module):
    """
    A complete embedding layer that processes an obfuscated image into a sequence of tokens
    compatible with a Vision Transformer (ViT) model, effectively replacing the model's
    own embedding layer.
    """
    
    def __init__(self, config: DeobfuscatorConfig) -> None:
        super().__init__()
        self.config = config

        image_size = config.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        patch_size = config.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, \
            'Input size must be divisible by patch size'

        self.image_size = image_size
        self.patch_size = patch_size
        
        h, w = self.image_size
        ph, pw = self.patch_size
        self.num_patches = (h // ph) * (w // pw)

        self.patch_embeddings = DeobfuscatorPatchEmbedding(config)

        if config.add_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        else:
            self.cls_token = None

        if config.add_position_embeddings:
            # Position embeddings for patch tokens + optional CLS token
            num_positions = self.num_patches + (1 if config.add_cls_token else 0)
            self.position_embeddings = nn.Parameter(torch.randn(1, num_positions, config.embed_dim))
        else:
            self.position_embeddings = None

        if config.num_extra_tokens > 0:
            self.detection_tokens = nn.Parameter(torch.randn(1, config.num_extra_tokens, config.embed_dim))
        else:
            self.detection_tokens = None

        # Compatibility with Hugging Face ViTModel's input type checking
        self.projection = SimpleNamespace(weight=torch.empty(0))
    
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        """
        if self.position_embeddings is None:
            return 0

        if self.config.add_cls_token:
            num_patches = embeddings.shape[1] - 1
            num_positions = self.position_embeddings.shape[1] - 1
            class_pos_embed = self.position_embeddings[:, :1]
            patch_pos_embed = self.position_embeddings[:, 1:]
        else:
            num_patches = embeddings.shape[1]
            num_positions = self.position_embeddings.shape[1]
            class_pos_embed = None
            patch_pos_embed = self.position_embeddings

        if num_patches == num_positions and (height, width) == self.image_size:
            return self.position_embeddings

        dim = embeddings.shape[-1]
        h0 = height // self.patch_size[0]
        w0 = width // self.patch_size[1]
        
        sqrt_num_positions = int(num_positions**0.5)
        if sqrt_num_positions * sqrt_num_positions != num_positions:
            raise ValueError(f"Cannot interpolate non-square position embeddings of size {num_positions}.")

        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        interpolated_patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h0, w0),
            mode="bicubic",
            align_corners=False,
        )
        interpolated_patch_pos_embed = interpolated_patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        if class_pos_embed is not None:
            return torch.cat((class_pos_embed, interpolated_patch_pos_embed), dim=1)
        return interpolated_patch_pos_embed

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the embedding layer.

        Args:
            pixel_values: Input tensor of shape (B, C, H, W).
            bool_masked_pos: Optional boolean tensor for masked patches (for MAE compatibility, not used here).
            interpolate_pos_encoding: Whether to interpolate position embeddings for different input image sizes.

        Returns:
            Output tensor of shape (B, sequence_length, embed_dim), ready for a transformer encoder.
        """
        B, C, H, W = pixel_values.shape

        if not interpolate_pos_encoding and (H, W) != self.image_size:
            raise ValueError(
                f"Input image size ({H}x{W}) doesn't match model"
                f" ({self.image_size[0]}x{self.image_size[1]})"
            )
        
        embeddings = self.patch_embeddings(pixel_values)
        
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if self.position_embeddings is not None:
            if interpolate_pos_encoding:
                position_embeddings = self.interpolate_pos_encoding(embeddings, H, W)
                embeddings = embeddings + position_embeddings
            else:
                embeddings = embeddings + self.position_embeddings

        if self.detection_tokens is not None:
            detection_tokens = self.detection_tokens.expand(B, -1, -1)
            embeddings = torch.cat((embeddings, detection_tokens), dim=1)

        return embeddings