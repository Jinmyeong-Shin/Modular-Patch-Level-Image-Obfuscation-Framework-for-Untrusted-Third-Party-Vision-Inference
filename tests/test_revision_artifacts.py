from types import SimpleNamespace

import torch

from vit_obfuscation.attacks.horio_baseline import (
    HorioPermutationConfig,
    HorioRestrictedPermutationObfuscator,
)
from vit_obfuscation.attacks.mi_fgsm import mi_fgsm_attack
from vit_obfuscation.attacks.lbfgs_inversion import _obfuscate_with_grad
from vit_obfuscation.obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from vit_obfuscation.outputs.manifest import build_manifest
from vit_obfuscation.tasks.captioning import _sentence_bleu
from vit_obfuscation.tasks.retrieval import _paired_recall


def test_mi_fgsm_attack_shape_and_bounds():
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=8,
        num_channels=3,
        patch_size=2,
        group_size=2,
    )
    original = torch.randn(2, 3, 8, 8).clamp(-1, 1)
    obfuscated = obfuscator(original)

    reconstructed = mi_fgsm_attack(
        obfuscator,
        obfuscated,
        iterations=2,
        step_size=0.01,
        num_restarts=1,
    )

    assert reconstructed.shape == original.shape
    assert reconstructed.min() >= -1
    assert reconstructed.max() <= 1


def test_gradient_obfuscation_respects_tanh_flag():
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=(1, 1),
        num_channels=1,
        patch_size=1,
        group_size=1,
        use_tanh=False,
    )
    obfuscator.obfuscation_weights.fill_(2.0)
    obfuscator.obfuscation_biases.fill_(0.5)
    obfuscator.channel_permutation[0] = 0

    x = torch.tensor([[[[1.0]]]], requires_grad=True)
    out = _obfuscate_with_grad(obfuscator, x)

    assert torch.equal(out, torch.tensor([[[[2.5]]]]))


def test_horio_restricted_permutation_inverse_roundtrip():
    baseline = HorioRestrictedPermutationObfuscator(
        image_size=(8, 8),
        num_channels=3,
        config=HorioPermutationConfig(
            patch_size=4,
            fixed_blocks=1,
            fixed_pixels=20,
            seed=7,
        ),
    )
    original = torch.linspace(-1, 1, steps=3 * 8 * 8).reshape(1, 3, 8, 8)

    encrypted = baseline(original)
    recovered = baseline.inverse(encrypted)

    assert torch.equal(recovered, original)
    summary = baseline.fixed_fraction_summary()
    assert summary["fixed_blocks"] == 1
    assert summary["fixed_pixels"] == 20


def test_manifest_contains_revision_reproducibility_fields():
    config = SimpleNamespace(
        name="demo",
        seed=42,
        model=SimpleNamespace(task="image_retrieval", hf_model_name_or_path="model"),
        dataset=SimpleNamespace(hf_dataset_name_or_path="dataset"),
    )

    manifest = build_manifest(
        config=config,
        result_file="outputs/revision_v3/demo_results.json",
        status="success",
    )

    assert manifest["experiment"] == "demo"
    assert manifest["status"] == "success"
    assert "environment" in manifest
    assert "git" in manifest


def test_paired_recall_diagonal_matches():
    similarity = torch.eye(4)
    assert _paired_recall(similarity, k=1, dim=1) == 1.0
    assert _paired_recall(similarity, k=1, dim=0) == 1.0


def test_sentence_bleu_rewards_overlap():
    good = _sentence_bleu("a red car on road", "a red car on the road", 1)
    bad = _sentence_bleu("blue ocean", "a red car on the road", 1)
    assert good > bad
