from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from run_adversarial_scenarios import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    _load_unit_images,
    _processor_image_size,
    _save_grid,
)
from vit_obfuscation.attacks.cyclegan_attack import (  # noqa: E402
    cyclegan_reconstruct,
    train_cyclegan_attack,
)
from vit_obfuscation.attacks.evaluate_attacks import compute_psnr, compute_ssim  # noqa: E402
from vit_obfuscation.attacks.horio_baseline import (  # noqa: E402
    HorioPermutationConfig,
    HorioRestrictedPermutationObfuscator,
)
from vit_obfuscation.attacks.side_channel import side_channel_analysis  # noqa: E402
from vit_obfuscation.attacks.vae_reconstruction import (  # noqa: E402
    train_vae_attack,
    vae_reconstruct,
)
from vit_obfuscation.config.experiment import ExperimentConfig  # noqa: E402
from vit_obfuscation.outputs.manifest import build_manifest  # noqa: E402


DEFAULT_CONFIG = "configs/experiments/vit_cifar10.yaml"


def _json_number(value: float) -> float | str:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    return float(value)


def _metric_row(name: str, reconstructed: torch.Tensor, original: torch.Tensor) -> dict:
    return {
        "attack": name,
        "ssim": _json_number(compute_ssim(reconstructed, original, data_range=2.0)),
        "psnr": _json_number(compute_psnr(reconstructed, original, data_range=2.0)),
        "mse": _json_number(F.mse_loss(reconstructed, original).item()),
    }


def _side_channel_dict(original: torch.Tensor, obfuscated: torch.Tensor) -> dict:
    side = side_channel_analysis(original, obfuscated)
    return {
        "frequency_correlation": side.frequency_correlation,
        "spatial_autocorrelation": side.spatial_autocorrelation,
        "histogram_kl_divergence": side.histogram_kl_divergence,
        "mutual_information": side.mutual_information,
    }


def _load_proposed_result(output_dir: Path, config_name: str) -> dict:
    path = output_dir / f"adversarial-{config_name}_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing proposed adversarial artifact: {path}. "
            "Run scripts/run_adversarial_scenarios.py first."
        )
    with path.open() as f:
        return json.load(f)


def _attack_by_name(proposed_result: dict) -> dict[str, dict]:
    return {row["attack"]: row for row in proposed_result.get("attacks", [])}


def _recover_global_permutation_from_chosen_plaintext(
    baseline: HorioRestrictedPermutationObfuscator,
    *,
    device: torch.device,
) -> torch.Tensor:
    total_values = baseline.num_channels * baseline.height * baseline.width
    marker = torch.arange(total_values, device=device).reshape(
        1,
        baseline.num_channels,
        baseline.height,
        baseline.width,
    )
    encrypted_marker = baseline(marker).reshape(-1).long()
    return encrypted_marker


def _invert_with_recovered_permutation(
    encrypted_images: torch.Tensor,
    recovered_mapping: torch.Tensor,
) -> torch.Tensor:
    flat_encrypted = encrypted_images.reshape(encrypted_images.shape[0], -1)
    flat_recovered = torch.empty_like(flat_encrypted)
    flat_recovered[:, recovered_mapping] = flat_encrypted
    return flat_recovered.reshape_as(encrypted_images)


def run_comparison(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    config = ExperimentConfig.from_yaml(args.config)
    torch.manual_seed(args.seed if args.seed is not None else config.seed)

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    image_size = _processor_image_size(config.model.hf_model_name_or_path)
    total_images = args.num_eval_images + args.num_train_images
    original = _load_unit_images(
        config.embedding_training.training_dataset,
        image_size,
        total_images,
    ).to(device)
    eval_original = original[: args.num_eval_images]
    train_original = original[args.num_eval_images :]

    horio = HorioRestrictedPermutationObfuscator(
        image_size=image_size,
        num_channels=3,
        config=HorioPermutationConfig(
            patch_size=args.patch_size,
            fixed_blocks=args.fixed_blocks,
            fixed_pixels=args.fixed_pixels,
            seed=args.horio_seed,
        ),
    ).to(device)
    horio.eval()

    with torch.no_grad():
        eval_horio = horio(eval_original)
        train_horio = horio(train_original)
        exact_key_inverse = horio.inverse(eval_horio)
        recovered_mapping = _recover_global_permutation_from_chosen_plaintext(
            horio,
            device=device,
        )
        chosen_plaintext_inverse = _invert_with_recovered_permutation(
            eval_horio,
            recovered_mapping,
        )

    vae = train_vae_attack(
        train_horio,
        train_original,
        epochs=args.vae_epochs,
    )
    horio_vae = vae_reconstruct(vae, eval_horio)

    learned_rows = [_metric_row("Adversarial VAE", horio_vae, eval_original)]
    grid_rows = [
        ("original", eval_original),
        ("Horio obfuscated", eval_horio),
        ("Horio VAE", horio_vae),
        ("Horio chosen-plaintext inverse", chosen_plaintext_inverse),
    ]

    if not args.skip_cyclegan:
        cyclegan = train_cyclegan_attack(
            train_horio,
            train_original,
            epochs=args.cyclegan_epochs,
        )
        horio_cyclegan = cyclegan_reconstruct(cyclegan, eval_horio)
        learned_rows.append(_metric_row("CycleGAN", horio_cyclegan, eval_original))
        grid_rows.insert(3, ("Horio CycleGAN", horio_cyclegan))

    proposed_result = _load_proposed_result(output_dir, config.name)
    proposed_attacks = _attack_by_name(proposed_result)

    horio_direct = _metric_row("obfuscated (no attack)", eval_horio, eval_original)
    horio_exact = _metric_row("exact-key inverse", exact_key_inverse, eval_original)
    horio_chosen = _metric_row(
        "chosen-plaintext recovered inverse",
        chosen_plaintext_inverse,
        eval_original,
    )
    horio_side = _side_channel_dict(eval_original, eval_horio)

    result_name = f"horio-adversary-comparison-{config.name}"
    result_path = output_dir / f"{result_name}_results.json"
    grid_path = output_dir / "figures" / f"{result_name}_grid.png"
    _save_grid(grid_rows, grid_path)

    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment": result_name,
        "base_experiment": config.name,
        "task": "adversarial_reconstruction_comparison",
        "tensor_domain": "rgb_resized_to_model_input_and_scaled_to_minus1_plus1",
        "image_size": list(image_size),
        "sample_counts": {
            "eval": args.num_eval_images,
            "attack_train": args.num_train_images,
            "chosen_plaintext_calibration": 1,
        },
        "horio_reference": {
            "citation_key": "horio2024privacypreservingvisiontransformerusing",
            "reported_setting": {
                "fixed_blocks": args.fixed_blocks,
                "fixed_pixels": args.fixed_pixels,
                "reported_cifar10_accuracy": args.reported_cifar10_accuracy,
            },
            "method_note": (
                "Restricted block and in-patch pixel permutation. "
                "The transform is lossless and exactly invertible when the secret "
                "permutations are known or recovered."
            ),
            "realized_fixed_fractions": horio.fixed_fraction_summary(),
        },
        "attack_config": {
            "vae_epochs": args.vae_epochs,
            "cyclegan_epochs": 0 if args.skip_cyclegan else args.cyclegan_epochs,
            "horio_seed": args.horio_seed,
        },
        "threat_models": {
            "black_box_passive_observation": {
                "description": (
                    "Adversary observes protected images but does not know the "
                    "secret obfuscation/permutation key."
                ),
                "proposed": {
                    "direct_obfuscated": proposed_attacks.get(
                        "obfuscated (no attack)"
                    ),
                    "side_channel": proposed_result.get("side_channel"),
                },
                "horio_restricted_permutation": {
                    "direct_obfuscated": horio_direct,
                    "side_channel": horio_side,
                },
            },
            "gray_box_paired_learned_inversion": {
                "description": (
                    "Adversary knows the method family and trains a reconstructor "
                    "from paired protected/clean samples drawn from the same image "
                    "distribution."
                ),
                "proposed": {
                    "Adversarial VAE": proposed_attacks.get("Adversarial VAE"),
                    "CycleGAN": proposed_attacks.get("CycleGAN"),
                },
                "horio_restricted_permutation": learned_rows,
            },
            "gray_box_chosen_plaintext": {
                "description": (
                    "For a pure permutation cipher, one calibration image with "
                    "unique pixel-channel values recovers the global inverse "
                    "permutation for this image size."
                ),
                "proposed": (
                    "Not directly comparable: the proposed transform is not a "
                    "lossless permutation, so exact-secret numerical inversion is "
                    "reported under the white-box threat model instead."
                ),
                "horio_restricted_permutation": horio_chosen,
            },
            "white_box_exact_secret": {
                "description": (
                    "Adversary knows the protected-image generation secret. "
                    "Horio can be exactly inverted; the proposed method is attacked "
                    "with exact-secret L-BFGS optimization."
                ),
                "proposed": {
                    "L-BFGS": proposed_attacks.get("L-BFGS"),
                    "MI-FGSM": proposed_attacks.get("MI-FGSM"),
                },
                "horio_restricted_permutation": horio_exact,
            },
        },
        "paper_interpretation": [
            (
                "Use L-BFGS as a conservative worst-case empirical attack, not as "
                "a formal proof of non-invertibility."
            ),
            (
                "The Horio high-accuracy restricted-permutation setting is useful "
                "for classification, but it remains a lossless permutation: exact "
                "key exposure or one chosen calibration image reconstructs the "
                "input perfectly."
            ),
            (
                "A defensible claim is that the proposed obfuscator retains lower "
                "visual recoverability under the tested practical attacks, while "
                "Horio-style encryption has a sharper key-compromise failure mode."
            ),
        ],
        "figure": str(grid_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with result_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)

    manifest_config = copy.deepcopy(config)
    manifest_config.name = result_name
    manifest = build_manifest(
        config=manifest_config,
        result_file=result_path,
        status="success",
    )
    manifest["attack_config"] = result["attack_config"]
    manifest_path = output_dir / f"{result_name}_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare proposed adversarial reconstruction artifacts with a Horio "
            "restricted-random-permutation baseline."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-eval-images", type=int, default=8)
    parser.add_argument("--num-train-images", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--fixed-blocks", type=int, default=120)
    parser.add_argument("--fixed-pixels", type=int, default=500)
    parser.add_argument("--reported-cifar10-accuracy", type=float, default=0.973)
    parser.add_argument("--horio-seed", type=int, default=42)
    parser.add_argument("--vae-epochs", type=int, default=8)
    parser.add_argument("--cyclegan-epochs", type=int, default=3)
    parser.add_argument("--skip-cyclegan", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = run_comparison(args)
    print(f"[done] wrote {result['experiment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
