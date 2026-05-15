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
    expected_types = ["vit", "clip", "yolos", "owlvit", "blip"]
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


def test_copy_frozen_params_skips_mismatched_position_embedding():
    """Should preserve target positional embedding when source embeddings mismatch."""

    class DummyEmbeddings(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_token = nn.Parameter(torch.randn(1, 4))
            self.position_embedding = nn.Embedding(10, 4)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = DummyEmbeddings()
            self.config = type("cfg", (), {"model_type": "vit"})()

    model = DummyModel()
    spec = EmbeddingSpec(
        embedding_path="embeddings",
        cls_token_attr="cls_token",
        position_embedding_attr="position_embedding",
    )
    adapter = ModelAdapter(model, spec)

    obf_embedding = ObfuscationEmbedding(
        image_size=32,
        num_channels=3,
        patch_size=16,
        embed_dim=4,
        num_extra_tokens=0,
    )

    # Source embedding has 10 positions, while target expects 5.
    adapter.copy_frozen_params(obf_embedding)

    assert obf_embedding.position_embedding.num_embeddings == 5
    assert obf_embedding.position_embedding.weight.shape == (5, 4)


def test_copy_frozen_params_accepts_batched_vit_position_parameter():
    """ViT position embeddings are stored as a [1, tokens, dim] parameter."""

    class DummyEmbeddings(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_token = nn.Parameter(torch.randn(1, 1, 4))
            self.position_embeddings = nn.Parameter(torch.randn(1, 5, 4))

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = DummyEmbeddings()
            self.config = type("cfg", (), {"model_type": "vit"})()

    model = DummyModel()
    adapter = ModelAdapter(model, _KNOWN_SPECS["vit"])
    obf_embedding = ObfuscationEmbedding(
        image_size=32,
        num_channels=3,
        patch_size=16,
        embed_dim=4,
        num_extra_tokens=0,
    )

    adapter.copy_frozen_params(obf_embedding)

    assert isinstance(obf_embedding.position_embedding, nn.Embedding)
    assert obf_embedding.position_embedding.weight.shape == (5, 4)
    assert torch.equal(
        obf_embedding.position_embedding.weight,
        model.embeddings.position_embeddings.squeeze(0),
    )


def test_copy_frozen_params_interpolates_yolos_position_embedding():
    """YOLOS rectangular source position embeddings should fit target image size."""

    class DummyInterpolation(nn.Module):
        def __init__(self, expected_tokens):
            super().__init__()
            self.expected_tokens = expected_tokens
            self.seen_size = None

        def forward(self, position_embeddings, image_size):
            self.seen_size = image_size
            dim = position_embeddings.shape[-1]
            values = torch.arange(self.expected_tokens * dim, dtype=torch.float32)
            return values.reshape(1, self.expected_tokens, dim)

    class DummyEmbeddings(torch.nn.Module):
        def __init__(self, expected_tokens):
            super().__init__()
            self.cls_token = nn.Parameter(torch.randn(1, 1, 4))
            self.detection_tokens = nn.Parameter(torch.randn(1, 2, 4))
            self.position_embeddings = nn.Parameter(torch.randn(1, 13, 4))
            self.interpolation = DummyInterpolation(expected_tokens)

    class DummyModel(torch.nn.Module):
        def __init__(self, expected_tokens):
            super().__init__()
            self.embeddings = DummyEmbeddings(expected_tokens)
            self.config = type("cfg", (), {"model_type": "yolos"})()

    obf_embedding = ObfuscationEmbedding(
        image_size=(32, 48),
        num_channels=3,
        patch_size=16,
        embed_dim=4,
        num_extra_tokens=2,
    )
    model = DummyModel(expected_tokens=obf_embedding.num_positions)
    spec = EmbeddingSpec(
        embedding_path="embeddings",
        cls_token_attr="cls_token",
        position_embedding_attr="position_embeddings",
        extra_tokens={"detection_tokens": "detection_tokens"},
    )
    adapter = ModelAdapter(model, spec)

    adapter.copy_frozen_params(obf_embedding)

    expected = torch.arange(obf_embedding.num_positions * 4, dtype=torch.float32)
    expected = expected.reshape(obf_embedding.num_positions, 4)
    assert model.embeddings.interpolation.seen_size == (32, 48)
    assert isinstance(obf_embedding.position_embedding, nn.Embedding)
    assert torch.equal(obf_embedding.position_embedding.weight, expected)


def test_get_num_extra_tokens_counts_multiple_extra_tokens():
    """Should count the total number of extra tokens from the original embedding."""

    class DummyEmbeddings(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_token = nn.Parameter(torch.randn(1, 4))
            self.position_embedding = nn.Embedding(5, 4)
            self.detection_tokens = nn.Parameter(torch.randn(1, 100, 4))

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = DummyEmbeddings()
            self.config = type("cfg", (), {"model_type": "yolos"})()

    model = DummyModel()
    spec = EmbeddingSpec(
        embedding_path="embeddings",
        cls_token_attr="cls_token",
        position_embedding_attr="position_embedding",
        extra_tokens={"detection_tokens": "detection_tokens"},
    )
    adapter = ModelAdapter(model, spec)

    assert adapter.get_num_extra_tokens() == 100


if __name__ == "__main__":
    test_resolve_attr()
    test_set_attr()
    test_known_specs_exist()
    test_embedding_spec_defaults()
    test_copy_frozen_params_skips_mismatched_position_embedding()
    test_copy_frozen_params_accepts_batched_vit_position_parameter()
    test_get_num_extra_tokens_counts_multiple_extra_tokens()
    print("All adapter tests passed!")
