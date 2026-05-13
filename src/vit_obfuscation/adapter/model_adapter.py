from __future__ import annotations

import copy
import functools
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

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
                target.position_embedding = copy.deepcopy(src_pos)
            elif isinstance(src_pos, nn.Parameter):
                target.position_embedding = nn.Parameter(src_pos.data.clone())
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

    def swap_to_obfuscation(self, obf_embedding: ObfuscationEmbedding) -> None:
        self.set_embedding_layer(obf_embedding)

    def swap_to_original(self) -> None:
        self.set_embedding_layer(self._original_embeddings)

    @contextmanager
    def obfuscation_mode(self, obf_embedding: ObfuscationEmbedding):
        """Context manager that temporarily swaps to obfuscation embedding."""
        self.swap_to_obfuscation(obf_embedding)
        try:
            yield
        finally:
            self.swap_to_original()
