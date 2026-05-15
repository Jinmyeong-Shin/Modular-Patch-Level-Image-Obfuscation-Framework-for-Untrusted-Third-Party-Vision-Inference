"""Run all experiments sequentially with resource management.

Designed to avoid OOM and disk space issues by:
- Running one experiment at a time with GPU cache clearing
- Monitoring disk usage between runs
- Skipping already-completed experiments
- Processing smallest datasets first
- Cleaning HF cache between model groups
"""

import gc
import json
import logging
import os
import shutil
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
for name in ["httpx", "httpcore", "filelock", "huggingface_hub"]:
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("experiment_runner")

OUTPUT_DIR = "./outputs/revision_v3"

# Ordered by model group and dataset size (smallest first)
# This ensures checkpoint reuse within model groups
EXPERIMENT_ORDER = [
    # Group 1: ViT-base (shared embedding checkpoint)
    "configs/experiments/vit_cifar10.yaml",  # CIFAR-10: 60K images, 10 classes
    "configs/experiments/vit_cifar100.yaml",  # CIFAR-100: 60K images, 100 classes
    "configs/experiments/vit_pathmnist.yaml",  # PathMNIST: ~100K, 9 classes
    "configs/experiments/vit_dermamnist.yaml",  # DermaMNIST: ~10K, 7 classes
    "configs/experiments/vit_bloodmnist.yaml",  # BloodMNIST: ~17K, 8 classes
    # Group 2: ViT-base ImageNet (same model, large dataset — separate for safety)
    "configs/experiments/vit_imagenet.yaml",  # ImageNet: 1.28M images
    # Group 3: CLIP
    "configs/experiments/clip_zero_shot.yaml",  # CLIP zero-shot
    # Revision-v3 task coverage additions
    "configs/experiments/mvtec_anomaly.yaml",  # industrial anomaly detection
    "configs/experiments/clip_image_retrieval.yaml",  # image-image retrieval
    "configs/experiments/clip_flickr30k_retrieval.yaml",  # image-text retrieval
    "configs/experiments/medical_minimsd_segmentation.yaml",  # binary medical segmentation
    "configs/experiments/blip_flickr30k_captioning.yaml",  # image captioning
    # Group 4: Object Detection
    "configs/experiments/yolos_coco.yaml",  # YOLOS object detection on COCO
    "configs/experiments/owlvit_coco.yaml",  # OWL-ViT zero-shot object detection on COCO
    # SKIPPED: ViT-Large highres (384x384 — high memory)
]


def check_disk_space(min_gb: float = 5.0) -> bool:
    """Check if enough disk space remains."""
    stat = shutil.disk_usage("/workspace")
    free_gb = stat.free / (1024**3)
    logger.info(f"Disk: {free_gb:.1f} GB free on /workspace")
    if free_gb < min_gb:
        logger.error(f"Low disk space: {free_gb:.1f} GB < {min_gb} GB minimum")
        return False
    return True


def check_gpu_memory():
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(
            f"GPU: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved, "
            f"{total:.1f} GB total"
        )


def clear_gpu():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def is_completed(config_path: str) -> bool:
    """Check if experiment already has saved results."""
    from vit_obfuscation.config.experiment import ExperimentConfig

    config = ExperimentConfig.from_yaml(config_path)
    result_path = os.path.join(OUTPUT_DIR, f"{config.name}_results.json")
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                data = json.load(f)
            if "error" not in data and "clean" in data and "obfuscated" in data:
                logger.info(f"SKIP (already completed): {config.name}")
                return True
        except Exception:
            pass
    return False


def run_single_experiment(config_path: str) -> dict | None:
    """Run a single experiment with full resource management."""
    from vit_obfuscation.config.experiment import ExperimentConfig
    from vit_obfuscation.runner.runner import ExperimentRunner
    from vit_obfuscation.outputs.manifest import write_manifest

    config = ExperimentConfig.from_yaml(config_path)
    config.output_dir = OUTPUT_DIR

    logger.info(f"\n{'=' * 70}")
    logger.info(f"EXPERIMENT: {config.name}")
    logger.info(f"  Model: {config.model.hf_model_name_or_path}")
    logger.info(f"  Task:  {config.model.task}")
    logger.info(f"  Dataset: {config.dataset.hf_dataset_name_or_path}")
    logger.info(f"{'=' * 70}")

    # Aggressively clear GPU before each experiment
    clear_gpu()
    check_gpu_memory()

    try:
        runner = ExperimentRunner(config)
        results = runner.run()
        del runner
        logger.info(f"SUCCESS: {config.name}")
        logger.info(f"  Clean:      {results.get('clean', {})}")
        logger.info(f"  Obfuscated: {results.get('obfuscated', {})}")
        return results
    except Exception as e:
        logger.error(f"FAILED: {config.name}: {e}", exc_info=True)
        error_result = {
            "experiment": config.name,
            "model": config.model.hf_model_name_or_path,
            "task": config.model.task,
            "error": str(e),
        }
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f"{config.name}_results.json"), "w") as f:
            json.dump(error_result, f, indent=2)
        write_manifest(
            config=config,
            output_dir=OUTPUT_DIR,
            result_file=os.path.join(OUTPUT_DIR, f"{config.name}_results.json"),
            status="error",
            error=str(e),
        )
        return error_result
    finally:
        # Always clear GPU after each experiment
        clear_gpu()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EXPERIMENT SUITE: Starting")
    logger.info(f"Total experiments: {len(EXPERIMENT_ORDER)}")
    logger.info("=" * 70)

    if not check_disk_space(5.0):
        logger.error("Aborting due to low disk space")
        sys.exit(1)

    all_results = []
    completed = 0
    skipped = 0
    failed = 0

    for i, config_path in enumerate(EXPERIMENT_ORDER, 1):
        logger.info(f"\n[{i}/{len(EXPERIMENT_ORDER)}] {config_path}")

        if not os.path.exists(config_path):
            logger.warning(f"Config not found: {config_path}, skipping")
            continue

        if is_completed(config_path):
            skipped += 1
            continue

        if not check_disk_space(3.0):
            logger.error("Stopping: disk space critically low")
            break

        result = run_single_experiment(config_path)
        if result and "error" not in result:
            completed += 1
        else:
            failed += 1

        all_results.append(result)

        # Clear memory between experiments
        clear_gpu()
        check_gpu_memory()

    # Save combined results
    combined_path = os.path.join(OUTPUT_DIR, "all_results.json")
    existing = []
    if os.path.exists(combined_path):
        try:
            with open(combined_path) as f:
                existing = json.load(f)
        except Exception:
            pass

    # Merge: update existing entries, add new ones
    existing_names = {r.get("experiment") for r in existing}
    for r in all_results:
        if r and r.get("experiment") not in existing_names:
            existing.append(r)
        elif r:
            existing = [
                r if e.get("experiment") == r.get("experiment") else e for e in existing
            ]

    with open(combined_path, "w") as f:
        json.dump(existing if existing else all_results, f, indent=2, default=str)

    logger.info(f"\n{'=' * 70}")
    logger.info(
        f"SUITE COMPLETE: {completed} succeeded, {skipped} skipped, {failed} failed"
    )
    logger.info(f"Results: {combined_path}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
