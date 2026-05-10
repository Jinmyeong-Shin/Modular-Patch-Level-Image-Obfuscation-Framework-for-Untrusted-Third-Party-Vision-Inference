import torch
import torch.nn as nn

from vit_obfuscation.adapter.model_adapter import ModelAdapter, _resolve_attr, _set_attr
from vit_obfuscation.adapter.registry import (
    _KNOWN_SPECS,
    EmbeddingSpec,
    get_embedding_spec,
)
from vit_obfuscation.embedding.embedding import ObfuscationEmbedding


def test_resolve_attr():
    """Should resolve dotted attribute paths."""

    class Inner:
        value = 42

    class Outer:
        inner = Inner()

    assert _resolve_attr(Outer(), "inner.value") == 42
    assert _resolve_attr(Outer(), "inner") is not None


def test_set_attr():
    """Should set attributes at dotted paths."""

    class Inner:
        value = 42

    class Outer:
        inner = Inner()

    obj = Outer()
    _set_attr(obj, "inner.value", 99)
    assert obj.inner.value == 99


def test_known_specs_exist():
    """Registry should have specs for known model types."""
    expected_types = ["vit", "clip", "yolos", "owlvit", "segformer", "clipseg"]
    for model_type in expected_types:
        assert model_type in _KNOWN_SPECS, f"Missing spec for {model_type}"


def test_embedding_spec_defaults():
    """EmbeddingSpec should have sensible defaults."""
    spec = EmbeddingSpec(embedding_path="embeddings")
    assert spec.cls_token_attr == "cls_token"
    assert spec.position_embedding_attr == "position_embeddings"
    assert spec.position_is_nn_embedding is True
    assert spec.extra_tokens == {}
    assert spec.is_hierarchical is False


if __name__ == "__main__":
    test_resolve_attr()
    test_set_attr()
    test_known_specs_exist()
    test_embedding_spec_defaults()
    print("All adapter tests passed!")
