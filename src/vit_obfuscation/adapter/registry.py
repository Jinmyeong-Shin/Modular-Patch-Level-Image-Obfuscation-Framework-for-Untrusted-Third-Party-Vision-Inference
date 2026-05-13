from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingSpec:
    """Describes how to find and interact with the vision embedding layer of a HF model."""

    embedding_path: str
    cls_token_attr: str | None = "cls_token"
    position_embedding_attr: str = "position_embeddings"
    position_is_nn_embedding: bool = True
    extra_tokens: dict[str, str] = field(default_factory=dict)
    vision_config_path: str = ""
    is_hierarchical: bool = False


# Known model registry keyed by model.config.model_type
_KNOWN_SPECS: dict[str, EmbeddingSpec] = {
    "vit": EmbeddingSpec(
        embedding_path="embeddings",
        cls_token_attr="cls_token",
        position_embedding_attr="position_embeddings",
        position_is_nn_embedding=True,
    ),
    "clip": EmbeddingSpec(
        embedding_path="vision_model.embeddings",
        cls_token_attr="class_embedding",
        position_embedding_attr="position_embedding",
        position_is_nn_embedding=True,
        vision_config_path="vision_config",
    ),
    "clip_vision_model": EmbeddingSpec(
        embedding_path="embeddings",
        cls_token_attr="class_embedding",
        position_embedding_attr="position_embedding",
        position_is_nn_embedding=True,
    ),
    "yolos": EmbeddingSpec(
        embedding_path="vit.embeddings",
        cls_token_attr="cls_token",
        position_embedding_attr="position_embeddings",
        position_is_nn_embedding=True,
        extra_tokens={"detection_tokens": "detection_tokens"},
    ),
    "owlvit": EmbeddingSpec(
        embedding_path="owlvit.vision_model.embeddings",
        cls_token_attr="class_embedding",
        position_embedding_attr="position_embedding",
        position_is_nn_embedding=True,
        vision_config_path="vision_config",
    ),
    "owlv2": EmbeddingSpec(
        embedding_path="owlv2.vision_model.embeddings",
        cls_token_attr="class_embedding",
        position_embedding_attr="position_embedding",
        position_is_nn_embedding=True,
        vision_config_path="vision_config",
    ),
    "beit": EmbeddingSpec(
        embedding_path="embeddings",
        cls_token_attr="cls_token",
        position_embedding_attr="position_embeddings",
        position_is_nn_embedding=True,
    ),
    "vit_mae": EmbeddingSpec(
        embedding_path="embeddings",
        cls_token_attr="cls_token",
        position_embedding_attr="position_embeddings",
        position_is_nn_embedding=True,
    ),
    "vit_msn": EmbeddingSpec(
        embedding_path="embeddings",
        cls_token_attr="cls_token",
        position_embedding_attr="position_embeddings",
        position_is_nn_embedding=True,
    ),
    "segformer": EmbeddingSpec(
        embedding_path="segformer.encoder.patch_embeddings",
        cls_token_attr=None,
        position_embedding_attr="",
        is_hierarchical=True,
    ),
    "clipseg": EmbeddingSpec(
        embedding_path="clip.vision_model.embeddings",
        cls_token_attr="class_embedding",
        position_embedding_attr="position_embedding",
        position_is_nn_embedding=True,
        vision_config_path="vision_config",
    ),
}


def register_embedding_spec(model_type: str, spec: EmbeddingSpec) -> None:
    """Register a custom EmbeddingSpec for a model type."""
    _KNOWN_SPECS[model_type] = spec


def get_embedding_spec(model: PreTrainedModel) -> EmbeddingSpec:
    """
    Get EmbeddingSpec for a model. Checks registry first, then falls back
    to introspection.
    """
    model_type = getattr(model.config, "model_type", None)
    if model_type and model_type in _KNOWN_SPECS:
        logger.info(f"Using registry spec for model_type='{model_type}'")
        return _KNOWN_SPECS[model_type]

    logger.info(f"Model type '{model_type}' not in registry, attempting introspection")
    return _discover_embedding(model)


def _discover_embedding(model: PreTrainedModel) -> EmbeddingSpec:
    """
    Introspect a HF model to find the vision embedding layer.
    Looks for modules with patch embedding sublayers, CLS tokens, and position embeddings.
    """
    candidates = []

    for name, module in model.named_modules():
        if name == "":
            continue

        has_patch_embed = False
        patch_embed_names = ["patch_embeddings", "patch_embedding", "proj"]
        for pe_name in patch_embed_names:
            child = getattr(module, pe_name, None)
            if child is not None and isinstance(child, nn.Module):
                has_patch_embed = True
                break

        if not has_patch_embed:
            continue

        # Check for CLS token
        cls_attr = None
        for attr_name in ["cls_token", "class_embedding"]:
            attr = getattr(module, attr_name, None)
            if attr is not None and isinstance(attr, (nn.Parameter, torch.Tensor)):
                cls_attr = attr_name
                break

        # Check for position embedding
        pos_attr = None
        pos_is_embedding = True
        for attr_name in ["position_embeddings", "position_embedding"]:
            attr = getattr(module, attr_name, None)
            if attr is not None:
                if isinstance(attr, nn.Embedding):
                    pos_attr = attr_name
                    pos_is_embedding = True
                    break
                elif isinstance(attr, nn.Parameter):
                    pos_attr = attr_name
                    pos_is_embedding = False
                    break

        # Check for extra tokens
        extra = {}
        for token_name in ["detection_tokens"]:
            if getattr(module, token_name, None) is not None:
                extra[token_name] = token_name

        # Prefer candidates with vision-related paths
        score = 0
        if cls_attr is not None:
            score += 2
        if pos_attr is not None:
            score += 2
        if "vision" in name.lower():
            score += 1
        # Penalize deep nesting
        score -= name.count(".")

        candidates.append((score, name, cls_attr, pos_attr, pos_is_embedding, extra))

    if not candidates:
        raise ValueError(
            f"Could not find vision embedding layer in model {type(model).__name__}. "
            "Please register an EmbeddingSpec manually using register_embedding_spec()."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, path, cls_attr, pos_attr, pos_is_embedding, extra = candidates[0]

    # Determine vision config path
    vision_config_path = ""
    if hasattr(model.config, "vision_config"):
        vision_config_path = "vision_config"

    spec = EmbeddingSpec(
        embedding_path=path,
        cls_token_attr=cls_attr,
        position_embedding_attr=pos_attr or "",
        position_is_nn_embedding=pos_is_embedding,
        extra_tokens=extra,
        vision_config_path=vision_config_path,
    )
    logger.info(f"Discovered embedding spec: {spec}")
    return spec
