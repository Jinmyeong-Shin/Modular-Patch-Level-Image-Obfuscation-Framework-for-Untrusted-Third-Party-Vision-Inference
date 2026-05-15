from __future__ import annotations

import copy
import functools
import logging
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)

from ..embedding.embedding import ObfuscationEmbedding
from .registry import EmbeddingSpec, get_embedding_spec


def _resolve_attr(obj, dotted_path: str):
    """Resolve a dotted attribute path like 'vit.embeddings'."""
    if not dotted_path:
        return obj
    return functools.reduce(getattr, dotted_path.split("."), obj)


def _set_attr(obj, dotted_path: str, value):
    """Set an attribute at a dotted path like 'vit.embeddings'."""
    parts = dotted_path.split(".")
    parent = functools.reduce(getattr, parts[:-1], obj)
    setattr(parent, parts[-1], value)


class ModelAdapter:
    """
    Generic adapter that handles embedding discovery, swapping, and frozen
    parameter copying for any ViT-based HuggingFace model.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        spec: EmbeddingSpec | None = None,
    ) -> None:
        self.model = model
        self.spec = spec or get_embedding_spec(model)
        self._original_embeddings = self.get_embedding_layer()

    def get_embedding_layer(self) -> nn.Module:
        return _resolve_attr(self.model, self.spec.embedding_path)

    def set_embedding_layer(self, embedding: nn.Module) -> None:
        _set_attr(self.model, self.spec.embedding_path, embedding)

    @property
    def original_embeddings(self) -> nn.Module:
        return self._original_embeddings

    def get_vision_config(self) -> PretrainedConfig:
        config = self.model.config
        if self.spec.vision_config_path:
            config = _resolve_attr(config, self.spec.vision_config_path)
        return config

    def copy_frozen_params(self, target: ObfuscationEmbedding) -> None:
        """Copy CLS token, position embeddings, and extra tokens from original to target."""
        src = self._original_embeddings

        # CLS token
        if self.spec.cls_token_attr is not None:
            src_cls = getattr(src, self.spec.cls_token_attr)
            if isinstance(src_cls, nn.Parameter):
                target.cls_embedding.data = src_cls.data.view(-1)[: target.embed_dim]
            else:
                target.cls_embedding.data = src_cls.data.view(-1)[: target.embed_dim]

        # Position embedding
        if self.spec.position_embedding_attr:
            src_pos = getattr(src, self.spec.position_embedding_attr, None)
            if src_pos is None:
                target.position_embedding = None
            elif isinstance(src_pos, nn.Embedding):
                if isinstance(target.position_embedding, nn.Embedding):
                    if (
                        src_pos.num_embeddings == target.position_embedding.num_embeddings
                        and src_pos.embedding_dim == target.position_embedding.embedding_dim
                    ):
                        target.position_embedding = copy.deepcopy(src_pos)
                    else:
                        logger.info(
                            "Skipping source position embedding copy because source and"
                            " target position embedding sizes differ: "
                            f"src={src_pos.num_embeddings}, target={target.position_embedding.num_embeddings}"
                        )
                elif isinstance(target.position_embedding, nn.Parameter):
                    if src_pos.weight.shape == target.position_embedding.shape:
                        target.position_embedding = nn.Parameter(src_pos.weight.data.clone())
                    else:
                        logger.info(
                            "Skipping source position embedding copy because source and"
                            " target position embedding shapes differ: "
                            f"src={src_pos.weight.shape}, target={target.position_embedding.shape}"
                        )
                else:
                    logger.info(
                        "Skipping source position embedding copy because target"
                        " position embedding type is unsupported"
                    )
            elif isinstance(src_pos, nn.Parameter):
                if isinstance(target.position_embedding, nn.Parameter):
                    if src_pos.data.shape == target.position_embedding.shape:
                        target.position_embedding = nn.Parameter(src_pos.data.clone())
                    elif (
                        src_pos.data.ndim == 3
                        and src_pos.data.shape[0] == 1
                        and src_pos.data.squeeze(0).shape
                        == target.position_embedding.shape
                    ):
                        target.position_embedding = nn.Parameter(src_pos.data.clone())
                    else:
                        logger.info(
                            "Skipping source position embedding copy because source and"
                            " target position embedding shapes differ: "
                            f"src={src_pos.data.shape}, target={target.position_embedding.shape}"
                        )
                elif isinstance(target.position_embedding, nn.Embedding):
                    if src_pos.data.shape == target.position_embedding.weight.shape:
                        target.position_embedding = nn.Embedding.from_pretrained(
                            src_pos.data.clone(), freeze=False
                        )
                    elif (
                        src_pos.data.ndim == 3
                        and src_pos.data.shape[0] == 1
                        and src_pos.data.squeeze(0).shape
                        == target.position_embedding.weight.shape
                    ):
                        target.position_embedding = nn.Embedding.from_pretrained(
                            src_pos.data.squeeze(0).clone(), freeze=False
                        )
                    else:
                        interpolated = self._interpolate_position_embeddings(
                            src, src_pos, target
                        )
                        if interpolated is not None:
                            target.position_embedding = nn.Embedding.from_pretrained(
                                interpolated.squeeze(0).clone(), freeze=False
                            )
                        else:
                            logger.info(
                                "Skipping source position embedding copy because source"
                                " and target position embedding shapes differ: "
                                f"src={src_pos.data.shape}, target={target.position_embedding.weight.shape}"
                            )
                else:
                    logger.info(
                        "Skipping source position embedding copy because target"
                        " position embedding type is unsupported"
                    )
            else:
                target.position_embedding = copy.deepcopy(src_pos)

        # Extra tokens (e.g., detection tokens)
        for target_attr, source_attr in self.spec.extra_tokens.items():
            src_tokens = getattr(src, source_attr)
            if target.extra_tokens is not None and isinstance(src_tokens, nn.Parameter):
                target.extra_tokens.data = src_tokens.data.clone()
            else:
                # For cases where extra_tokens shape doesn't match, deep copy
                setattr(target, "extra_tokens", copy.deepcopy(src_tokens))

    def _interpolate_position_embeddings(
        self,
        source_embeddings: nn.Module,
        source_position_embeddings: nn.Parameter,
        target: ObfuscationEmbedding,
    ) -> torch.Tensor | None:
        interpolation = getattr(source_embeddings, "interpolation", None)
        if interpolation is None:
            return None

        with torch.no_grad():
            try:
                interpolated = interpolation(
                    source_position_embeddings.data,
                    tuple(target.image_size),
                )
            except Exception as exc:
                logger.info(
                    "Skipping source position embedding interpolation because it"
                    f" failed: {exc}"
                )
                return None

        if interpolated.ndim == 2:
            interpolated = interpolated.unsqueeze(0)
        expected_shape = (1, target.num_positions, target.embed_dim)
        if tuple(interpolated.shape) != expected_shape:
            logger.info(
                "Skipping source position embedding interpolation because output"
                f" shape differs: src={source_position_embeddings.data.shape}, "
                f"interpolated={interpolated.shape}, expected={expected_shape}"
            )
            return None

        logger.info(
            "Interpolated source position embeddings from "
            f"{tuple(source_position_embeddings.data.shape)} to {expected_shape}"
        )
        return interpolated

    def swap_to_obfuscation(self, obf_embedding: ObfuscationEmbedding) -> None:
        self.set_embedding_layer(obf_embedding)

    def swap_to_original(self) -> None:
        self.set_embedding_layer(self._original_embeddings)

    def get_num_extra_tokens(self) -> int:
        """Return the total number of extra tokens defined in the original embeddings."""
        total = 0
        for source_attr in self.spec.extra_tokens.values():
            src_tokens = getattr(self._original_embeddings, source_attr, None)
            if src_tokens is None:
                continue
            if isinstance(src_tokens, torch.Tensor):
                if src_tokens.ndim == 3:
                    total += src_tokens.shape[1]
                elif src_tokens.ndim == 2:
                    total += src_tokens.shape[0]
            elif isinstance(src_tokens, nn.Module):
                weight = getattr(src_tokens, "weight", None)
                if isinstance(weight, torch.Tensor):
                    if weight.ndim == 3:
                        total += weight.shape[1]
                    elif weight.ndim == 2:
                        total += weight.shape[0]
        return total

    @contextmanager
    def obfuscation_mode(self, obf_embedding: ObfuscationEmbedding):
        """Context manager that temporarily swaps to obfuscation embedding."""
        self.swap_to_obfuscation(obf_embedding)
        try:
            yield
        finally:
            self.swap_to_original()
