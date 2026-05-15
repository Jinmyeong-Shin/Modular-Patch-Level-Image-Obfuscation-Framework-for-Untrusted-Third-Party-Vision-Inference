import torch

from vit_obfuscation.obfuscation.obfuscator import ChannelWisePatchLevelObfuscator


def test_obfuscator_output_shape():
    """Output shape should match input shape."""
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=224,
        num_channels=3,
        patch_size=14,
        group_size=10,
    )
    x = torch.randn(2, 3, 224, 224)
    out = obfuscator(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_obfuscator_output_bounded():
    """Output values should be in [-1, 1] due to tanh."""
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=224,
        num_channels=3,
        patch_size=14,
        group_size=10,
    )
    x = torch.randn(2, 3, 224, 224)
    out = obfuscator(x)
    assert out.min() >= -1.0, f"Min value {out.min()} below -1"
    assert out.max() <= 1.0, f"Max value {out.max()} above 1"


def test_obfuscator_deterministic():
    """Same input should produce same output (buffers are fixed)."""
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=224,
        num_channels=3,
        patch_size=14,
        group_size=10,
    )
    x = torch.randn(1, 3, 224, 224)
    out1 = obfuscator(x)
    out2 = obfuscator(x)
    assert torch.allclose(out1, out2), "Obfuscator should be deterministic"


def test_obfuscator_different_inputs():
    """Different inputs should produce different outputs."""
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=224,
        num_channels=3,
        patch_size=14,
        group_size=10,
    )
    x1 = torch.randn(1, 3, 224, 224)
    x2 = torch.randn(1, 3, 224, 224)
    out1 = obfuscator(x1)
    out2 = obfuscator(x2)
    assert not torch.allclose(out1, out2), (
        "Different inputs should give different outputs"
    )


def test_obfuscator_non_square():
    """Should work with non-square images."""
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=(224, 336),
        num_channels=3,
        patch_size=14,
        group_size=10,
    )
    x = torch.randn(1, 3, 224, 336)
    out = obfuscator(x)
    assert out.shape == x.shape


def test_obfuscator_does_not_apply_patch_permutation_by_default():
    """Default behavior should match the old PoC: patch permutations are inert."""
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=(1, 4),
        num_channels=1,
        patch_size=1,
        group_size=1,
        use_tanh=False,
    )
    obfuscator.obfuscation_weights.fill_(1.0)
    obfuscator.obfuscation_biases.zero_()
    obfuscator.patch_permutations[0] = torch.tensor([3, 2, 1, 0])
    obfuscator.channel_permutation[0] = 0

    x = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    out = obfuscator(x)

    assert torch.equal(out, x)


def test_obfuscator_applies_patch_permutation_when_enabled():
    """Patch permutations should move transformed patches only when enabled."""
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=(1, 4),
        num_channels=1,
        patch_size=1,
        group_size=1,
        use_tanh=False,
        apply_patch_permutation=True,
    )
    obfuscator.obfuscation_weights.fill_(1.0)
    obfuscator.obfuscation_biases.zero_()
    obfuscator.patch_permutations[0] = torch.tensor([3, 2, 1, 0])
    obfuscator.channel_permutation[0] = 0

    x = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    out = obfuscator(x)

    assert torch.equal(out, torch.tensor([[[[3.0, 2.0, 1.0, 0.0]]]]))


if __name__ == "__main__":
    test_obfuscator_output_shape()
    test_obfuscator_output_bounded()
    test_obfuscator_deterministic()
    test_obfuscator_different_inputs()
    test_obfuscator_non_square()
    test_obfuscator_does_not_apply_patch_permutation_by_default()
    test_obfuscator_applies_patch_permutation_when_enabled()
    print("All obfuscator tests passed!")
