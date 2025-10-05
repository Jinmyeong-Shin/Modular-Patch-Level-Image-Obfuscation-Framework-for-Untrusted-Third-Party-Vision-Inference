from types import SimpleNamespace
import math
from typing import Optional

import torch
import torch.nn as nn

class ObfuscationPatchEmbedding(nn.Module):
    """
    An embedding layer that processes an obfuscated image into a sequence of tokens
    compatible with a Vision Transformer (ViT) model.

    This layer performs three main vectorized operations:
    1.  **Patch Decoding**: Each patch from the input image (across all channels) is
        transformed into a higher-dimensional embedding. A unique linear transformation
        is learned for each patch position.
    2.  **Channel Merging**: The embeddings from the different input channels (e.g., R, G, B)
        for each patch are merged into a single embedding via a learned linear combination.
        Again, a unique combination is learned for each patch position.

    The entire process is implemented using efficient, vectorized tensor operations
    (primarily `torch.einsum`) to replace slow Python loops, making it suitable for
    high-performance training and inference.
    """
    
    def __init__(
        self,
        image_size: int | tuple[int, int],
        num_channels: int,
        patch_size: int,
        embed_dim: int,
        hidden_ratio: float = 4.0,
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
        self.embed_dim = embed_dim

        H, W = self.image_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        patch_dim = patch_size * patch_size

        # Parameters for patch decoding. This is equivalent to `num_patches` independent
        # nn.Linear(patch_dim, embed_dim) layers.
        # Shape: (num_patches, embed_dim, patch_dim)
        self.decode_weights = nn.Parameter(torch.empty(self.num_patches, embed_dim, patch_dim))
        self.decode_biases = nn.Parameter(torch.empty(self.num_patches, embed_dim))

        # Parameters for channel merging. This is equivalent to `num_patches` independent
        # nn.Linear(num_channels, 1) layers.
        # Shape: (num_patches, num_channels)
        self.merge_weights = nn.Parameter(torch.empty(self.num_patches, num_channels))
        self.merge_biases = nn.Parameter(torch.empty(self.num_patches))

        nn.init.normal_(self.decode_weights, std=0.01)
        nn.init.normal_(self.merge_weights, std=0.01)

        nn.init.zeros_(self.decode_biases)
        nn.init.zeros_(self.merge_biases)

        # Create a dummy projection attribute for compatibility with the original ViT model.
        # This is not a real layer, and it will not be saved with the model's state_dict.
        # It only exists to provide a `.weight.dtype` attribute, which the Hugging Face
        # ViTModel forward pass expects for input tensor type checking. By using
        # SimpleNamespace, it is not registered as a submodule.
        self.projection = SimpleNamespace(weight=torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels}, got {C}."
            )

        ps = self.patch_size
        Nh, Nw = H // ps, W // ps
        num_patches = Nh * Nw

        # 1. Vectorized Patch Extraction
        # (B, C, H, W) -> (B, C, num_patches, ps*ps)
        x_patched = x.view(B, C, Nh, ps, Nw, ps).permute(0, 1, 2, 4, 3, 5).reshape(B, C, num_patches, ps * ps)

        # 2. Vectorized Patch Decoding
        # Applies a different linear layer (patch_dim -> embed_dim) to each patch position.
        # 'bcnp,nfp->bcnf': (B, C, num_patches, patch_dim) @ (num_patches, embed_dim, patch_dim).T -> (B, C, num_patches, embed_dim)
        decoded_patches = torch.einsum('bcnp,nfp->bcnf', x_patched, self.decode_weights) + self.decode_biases.view(1, 1, num_patches, -1)
        
        # 3. Vectorized Channel Merging
        # Applies a different linear combination (num_channels -> 1) to each patch's embedding.
        # Permute to (B, num_patches, embed_dim, C) to make channel dimension last.
        x_permuted = decoded_patches.permute(0, 2, 3, 1)

        # 'bnec,nc->bne': (B, num_patches, embed_dim, C) * (num_patches, C) -> sum over C -> (B, num_patches, embed_dim)
        return torch.einsum('bnec,nc->bne', x_permuted, self.merge_weights) + self.merge_biases.view(1, num_patches, -1)

class ObfuscationEmbedding(nn.Module):
    
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

        assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0, \
            'Input size must be divisible by patch size'

        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        H, W = self.image_size
        self.num_patches = (H // patch_size) * (W // patch_size)

        self.cls_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = ObfuscationPatchEmbedding(
            image_size=self.image_size,
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim
        )
        self.patch_embeddings = self.patch_embedding # for origianl vit-based model

        # Parameters for position embedding.
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, embed_dim)
        self.register_buffer('position_ids', torch.arange(self.num_positions).expand((1, -1)), persistent=False)

        # Competibility with YOLOS
        self.detection_tokens = None
    
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

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
            bool_masked_pos: Optional boolean tensor indicating masked patches. This is for compatibility with the original ViT model and is not used.
            interpolate_pos_encoding: Whether to interpolate position embeddings for different input image sizes.

        Returns:
            Output tensor of shape (B, num_patches + 1, embed_dim), ready for a ViT.
        """
        B, C, H, W = pixel_values.shape

        if not interpolate_pos_encoding:
            if (H, W) != self.image_size:
                raise ValueError(
                    f"Input image size ({H}x{W}) doesn't match model"
                    f" ({self.image_size[0]}x{self.image_size[1]})"
                )
        x_out = self.patch_embedding(pixel_values)
        
        cls_embeds = self.cls_embedding.expand(B, 1, -1)
        x_out = torch.concat([cls_embeds, x_out], dim=1)

        if self.detection_tokens is not None:
            detection_tokens = self.detection_tokens.expand(B, -1, -1)
            x_out = torch.concat([x_out, detection_tokens], dim=1)

        if interpolate_pos_encoding:
            x_out = x_out + self.interpolate_pos_encoding(x_out, H, W)
        else:
            if isinstance(self.position_embedding, nn.Embedding):
                x_out = x_out + self.position_embedding(self.position_ids)
            elif isinstance(self.position_embedding, nn.Parameter):
                x_out = x_out + self.position_embedding
            else:
                raise RuntimeError('Unknown position embedding type')

        return x_out
