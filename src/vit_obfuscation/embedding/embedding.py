from typing import Optional

import torch
import torch.nn as nn

from .patch_embedding import ObfuscationPatchEmbedding


class ObfuscationEmbedding(nn.Module):
    """
    Complete embedding replacement for ViT models processing obfuscated images.
    Combines patch embedding with CLS token and position embeddings.

    Supports extra tokens (e.g., YOLOS detection tokens) via num_extra_tokens.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int],
        num_channels: int,
        patch_size: int,
        embed_dim: int,
        num_extra_tokens: int = 0,
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
        self.num_extra_tokens = num_extra_tokens

        H, W = self.image_size
        self.num_patches = (H // patch_size) * (W // patch_size)

        self.cls_embedding = nn.Parameter(torch.randn(embed_dim))

        self.patch_embedding = ObfuscationPatchEmbedding(
            image_size=self.image_size,
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
        )
        # Alias for HF ViT compatibility
        self.patch_embeddings = self.patch_embedding

        # Position embedding: 1 (CLS) + num_patches + num_extra_tokens
        self.num_positions = self.num_patches + 1 + num_extra_tokens
        self.position_embedding = nn.Embedding(self.num_positions, embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

        # Extra tokens (e.g., YOLOS detection tokens) - set by adapter
        if num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(
                torch.zeros(1, num_extra_tokens, embed_dim)
            )
        else:
            self.extra_tokens = None

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        if self.extra_tokens is not None:
            num_patches -= self.num_extra_tokens
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1
        if self.extra_tokens is not None:
            num_positions -= self.num_extra_tokens

        if (
            not torch.jit.is_tracing()
            and num_patches == num_positions
            and height == width
        ):
            return self.position_embedding(self.position_ids)

        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1 : 1 + self.num_patches]

        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        sqrt_num_positions = int(self.num_patches**0.5)

        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        result = torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        if self.extra_tokens is not None:
            extra_pos_embed = position_embedding[:, -self.num_extra_tokens :]
            result = torch.cat((result, extra_pos_embed), dim=1)
        return result

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        B, C, H, W = pixel_values.shape

        if not interpolate_pos_encoding:
            if (H, W) != self.image_size:
                raise ValueError(
                    f"Input image size ({H}x{W}) doesn't match model"
                    f" ({self.image_size[0]}x{self.image_size[1]})"
                )

        x_out = self.patch_embedding(pixel_values)

        cls_embeds = self.cls_embedding.expand(B, 1, -1)
        x_out = torch.cat([cls_embeds, x_out], dim=1)

        if self.extra_tokens is not None:
            extra = self.extra_tokens.expand(B, -1, -1)
            x_out = torch.cat([x_out, extra], dim=1)

        if interpolate_pos_encoding:
            x_out = x_out + self.interpolate_pos_encoding(x_out, H, W)
        else:
            if self.position_embedding is None:
                pass  # Model uses relative position bias (e.g., BEiT)
            elif isinstance(self.position_embedding, nn.Embedding):
                x_out = x_out + self.position_embedding(self.position_ids)
            elif isinstance(self.position_embedding, nn.Parameter):
                x_out = x_out + self.position_embedding
            else:
                raise RuntimeError("Unknown position embedding type")

        return x_out
