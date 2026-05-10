import torch

from vit_obfuscation.embedding.embedding import ObfuscationEmbedding
from vit_obfuscation.embedding.patch_embedding import ObfuscationPatchEmbedding


def test_patch_embedding_output_shape():
    """Patch embedding should produce (B, num_patches, embed_dim)."""
    pe = ObfuscationPatchEmbedding(
        image_size=224,
        num_channels=3,
        patch_size=16,
        embed_dim=768,
    )
    x = torch.randn(2, 3, 224, 224)
    out = pe(x)
    expected_patches = (224 // 16) ** 2  # 196
    assert out.shape == (2, expected_patches, 768), f"Got {out.shape}"


def test_embedding_output_shape():
    """Full embedding should produce (B, num_patches + 1, embed_dim) with CLS."""
    emb = ObfuscationEmbedding(
        image_size=224,
        num_channels=3,
        patch_size=16,
        embed_dim=768,
    )
    x = torch.randn(2, 3, 224, 224)
    out = emb(x)
    expected_seq_len = (224 // 16) ** 2 + 1  # 197
    assert out.shape == (2, expected_seq_len, 768), f"Got {out.shape}"


def test_embedding_with_extra_tokens():
    """Extra tokens (e.g., detection tokens) should extend sequence length."""
    emb = ObfuscationEmbedding(
        image_size=224,
        num_channels=3,
        patch_size=16,
        embed_dim=768,
        num_extra_tokens=100,
    )
    x = torch.randn(2, 3, 224, 224)
    out = emb(x)
    expected_seq_len = (224 // 16) ** 2 + 1 + 100  # 297
    assert out.shape == (2, expected_seq_len, 768), f"Got {out.shape}"


def test_embedding_gradient_flow():
    """Gradients should flow through the patch embedding."""
    emb = ObfuscationEmbedding(
        image_size=224,
        num_channels=3,
        patch_size=16,
        embed_dim=768,
    )
    x = torch.randn(2, 3, 224, 224)
    out = emb(x)
    loss = out.sum()
    loss.backward()

    assert emb.patch_embedding.decode_weights.grad is not None
    assert emb.patch_embedding.merge_weights.grad is not None


if __name__ == "__main__":
    test_patch_embedding_output_shape()
    test_embedding_output_shape()
    test_embedding_with_extra_tokens()
    test_embedding_gradient_flow()
    print("All embedding tests passed!")
