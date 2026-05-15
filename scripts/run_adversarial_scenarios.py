from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from vit_obfuscation.attacks.evaluate_attacks import evaluate_all_attacks
from vit_obfuscation.attacks.side_channel import side_channel_analysis
from vit_obfuscation.config.experiment import ExperimentConfig
from vit_obfuscation.datasets.loader import load_embedding_training_dataset
from vit_obfuscation.obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from vit_obfuscation.outputs.manifest import build_manifest
from vit_obfuscation.runner.runner import _checkpoint_key


DEFAULT_OUTPUT_DIR = Path("outputs/revision_v3")
DEFAULT_CONFIGS = ["configs/experiments/vit_cifar10.yaml"]


def _processor_image_size(model_name: str) -> tuple[int, int]:
    processor = AutoProcessor.from_pretrained(model_name)
    image_processor = getattr(processor, "image_processor", processor)
    size = getattr(image_processor, "size", None)

    height = width = None
    if isinstance(size, dict) or hasattr(size, "get"):
        height = size.get("height") or size.get("shortest_edge")
        width = size.get("width") or size.get("shortest_edge")
    elif size is not None:
        height = getattr(size, "height", None) or getattr(size, "shortest_edge", None)
        width = getattr(size, "width", None) or getattr(size, "shortest_edge", None)

    if height is None or width is None:
        height = width = 224
    return int(height), int(width)


def _load_obfuscator(config: ExperimentConfig, output_dir: Path) -> tuple[
    ChannelWisePatchLevelObfuscator,
    Path,
    tuple[int, int],
]:
    image_size = _processor_image_size(config.model.hf_model_name_or_path)
    obfuscator = ChannelWisePatchLevelObfuscator(
        image_size=image_size,
        num_channels=3,
        patch_size=config.obfuscation.patch_size,
        group_size=config.obfuscation.group_size,
        apply_patch_permutation=config.obfuscation.apply_patch_permutation,
    )

    checkpoint_path = output_dir / "checkpoints" / f"{_checkpoint_key(config)}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing obfuscator checkpoint: {checkpoint_path}. "
            "Run the corresponding performance experiment first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    obfuscator.load_state_dict(checkpoint["obfuscator"])
    return obfuscator, checkpoint_path, image_size


def _image_to_unit_tensor(image: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(resized).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _load_unit_images(
    dataset_name: str,
    image_size: tuple[int, int],
    count: int,
) -> torch.Tensor:
    dataset = load_embedding_training_dataset(dataset_name)
    images = [_image_to_unit_tensor(dataset[i]["image"], image_size) for i in range(count)]
    return torch.stack(images)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu().clamp(-1, 1)
    image = ((image + 1.0) * 127.5).round().to(torch.uint8)
    array = image.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _save_grid(
    rows: list[tuple[str, torch.Tensor]],
    path: Path,
    *,
    max_images: int = 4,
) -> None:
    if not rows:
        return
    images_per_row = min(max_images, rows[0][1].shape[0])
    cell_w, cell_h = _tensor_to_pil(rows[0][1][0]).size
    label_h = 22
    canvas = Image.new(
        "RGB",
        (cell_w * images_per_row, (cell_h + label_h) * len(rows)),
        "white",
    )

    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
    except Exception:
        draw = None

    for row_idx, (label, tensors) in enumerate(rows):
        y0 = row_idx * (cell_h + label_h)
        if draw is not None:
            draw.text((4, y0 + 3), label, fill=(0, 0, 0))
        for col_idx in range(images_per_row):
            canvas.paste(_tensor_to_pil(tensors[col_idx]), (col_idx * cell_w, y0 + label_h))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _result_rows(results) -> tuple[list[dict], list[tuple[str, torch.Tensor]]]:
    rows = []
    image_rows = []
    for result in results:
        rows.append(
            {
                "attack": result.attack_name,
                "ssim": result.ssim,
                "psnr": result.psnr,
                "mse": result.mse,
            }
        )
        image_rows.append((result.attack_name, result.reconstructed))
    return rows, image_rows


def run_adversarial_config(
    config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    config = ExperimentConfig.from_yaml(str(config_path))
    config.output_dir = str(output_dir)
    torch.manual_seed(args.seed if args.seed is not None else config.seed)

    obfuscator, checkpoint_path, image_size = _load_obfuscator(config, output_dir)
    total_images = args.num_eval_images + args.num_train_images
    original = _load_unit_images(
        config.embedding_training.training_dataset,
        image_size,
        total_images,
    )

    eval_original = original[: args.num_eval_images]
    train_original = original[args.num_eval_images :]

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    obfuscator = obfuscator.to(device).eval()
    eval_original = eval_original.to(device)
    train_original = train_original.to(device)

    with torch.no_grad():
        eval_obfuscated = obfuscator(eval_original)
        train_obfuscated = obfuscator(train_original)

    attack_results = evaluate_all_attacks(
        obfuscator=obfuscator,
        original_images=eval_original,
        obfuscated_images=eval_obfuscated,
        attack_train_original_images=train_original,
        attack_train_obfuscated_images=train_obfuscated,
        vae_epochs=args.vae_epochs,
        cyclegan_epochs=args.cyclegan_epochs,
        mi_fgsm_steps=args.mi_steps,
        lbfgs_iterations=args.lbfgs_iterations,
        lbfgs_restarts=args.lbfgs_restarts,
        data_range=2.0,
        clip_min=-1.0,
        clip_max=1.0,
        run_cyclegan=not args.skip_cyclegan,
    )

    side = side_channel_analysis(eval_original, eval_obfuscated)
    attack_rows, image_rows = _result_rows(attack_results)
    image_rows.insert(0, ("original", eval_original))

    result_name = f"adversarial-{config.name}"
    result_path = output_dir / f"{result_name}_results.json"
    grid_path = output_dir / "figures" / f"{result_name}_grid.png"
    _save_grid(image_rows, grid_path)

    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment": result_name,
        "base_experiment": config.name,
        "model": config.model.hf_model_name_or_path,
        "task": "adversarial_reconstruction",
        "checkpoint": str(checkpoint_path),
        "tensor_domain": "rgb_resized_to_model_input_and_scaled_to_minus1_plus1",
        "image_size": list(image_size),
        "sample_counts": {
            "eval": args.num_eval_images,
            "attack_train": args.num_train_images,
        },
        "attack_config": {
            "mi_steps": args.mi_steps,
            "lbfgs_iterations": args.lbfgs_iterations,
            "lbfgs_restarts": args.lbfgs_restarts,
            "vae_epochs": args.vae_epochs,
            "cyclegan_epochs": 0 if args.skip_cyclegan else args.cyclegan_epochs,
        },
        "attacks": attack_rows,
        "side_channel": {
            "frequency_correlation": side.frequency_correlation,
            "spatial_autocorrelation": side.spatial_autocorrelation,
            "histogram_kl_divergence": side.histogram_kl_divergence,
            "mutual_information": side.mutual_information,
        },
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
        description="Run revision-v3 adversarial reconstruction and side-channel benchmarks."
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Experiment config whose obfuscator checkpoint should be attacked.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-eval-images", type=int, default=8)
    parser.add_argument("--num-train-images", type=int, default=32)
    parser.add_argument("--mi-steps", type=int, default=60)
    parser.add_argument("--lbfgs-iterations", type=int, default=80)
    parser.add_argument("--lbfgs-restarts", type=int, default=2)
    parser.add_argument("--vae-epochs", type=int, default=8)
    parser.add_argument("--cyclegan-epochs", type=int, default=3)
    parser.add_argument("--skip-cyclegan", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_paths = [Path(path) for path in (args.configs or DEFAULT_CONFIGS)]

    results = []
    for config_path in config_paths:
        print(f"[run] adversarial benchmark for {config_path}")
        results.append(run_adversarial_config(config_path, output_dir, args))
        print(f"[ok] {results[-1]['experiment']}")

    combined_path = output_dir / "adversarial_results.json"
    with combined_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[done] wrote {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
