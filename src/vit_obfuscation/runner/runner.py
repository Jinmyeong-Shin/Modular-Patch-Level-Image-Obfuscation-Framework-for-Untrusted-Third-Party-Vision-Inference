from __future__ import annotations

import json
import logging
import os
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForObjectDetection, AutoProcessor

from ..adapter.model_adapter import ModelAdapter
from ..adapter.registry import get_embedding_spec
from ..config.experiment import ExperimentConfig
from ..embedding.embedding import ObfuscationEmbedding
from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from ..outputs.manifest import write_manifest
from .embedding_trainer import EmbeddingTrainer

logger = logging.getLogger(__name__)

_DETECTION_TASKS = {"object_detection", "zero_shot_object_detection"}
_AUTO_MODEL_TASKS = {
    "classification",
    "zero_shot_classification",
    "image_retrieval",
    "image_text_retrieval",
    "anomaly_detection",
    "medical_segmentation",
}

_TASK_REGISTRY = {
    "classification": "vit_obfuscation.tasks.classification.ClassificationTask",
    "zero_shot_classification": "vit_obfuscation.tasks.classification.ZeroShotClassificationTask",
    "object_detection": "vit_obfuscation.tasks.object_detection.ObjectDetectionTask",
    "zero_shot_object_detection": "vit_obfuscation.tasks.object_detection.ZeroShotObjectDetectionTask",
    "anomaly_detection": "vit_obfuscation.tasks.anomaly.AnomalyDetectionTask",
    "image_retrieval": "vit_obfuscation.tasks.retrieval.ImageRetrievalTask",
    "image_text_retrieval": "vit_obfuscation.tasks.retrieval.ImageTextRetrievalTask",
    "medical_segmentation": "vit_obfuscation.tasks.medical_segmentation.MedicalBinarySegmentationTask",
    "image_captioning": "vit_obfuscation.tasks.captioning.ImageCaptioningTask",
}


def _import_task_class(task_type: str):
    if task_type not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task type '{task_type}'. Available: {list(_TASK_REGISTRY.keys())}"
        )
    module_path, class_name = _TASK_REGISTRY[task_type].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_model(config: ExperimentConfig):
    task = config.model.task
    model_name = config.model.hf_model_name_or_path

    if task in _DETECTION_TASKS:
        if task == "zero_shot_object_detection":
            from transformers import OwlViTForObjectDetection

            return OwlViTForObjectDetection.from_pretrained(model_name)
        return AutoModelForObjectDetection.from_pretrained(model_name)
    elif task == "image_captioning":
        try:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq.from_pretrained(model_name)
        except Exception:
            from transformers import BlipForConditionalGeneration

            return BlipForConditionalGeneration.from_pretrained(model_name)
    elif task in _AUTO_MODEL_TASKS:
        return AutoModel.from_pretrained(model_name)
    else:
        return AutoModel.from_pretrained(model_name)


def _checkpoint_key(config: ExperimentConfig) -> str:
    """Generate a unique key for obfuscation checkpoint based on model + obfuscation params."""
    model_name = config.model.hf_model_name_or_path.replace("/", "_")
    permutation_tag = (
        "_pperm1" if config.obfuscation.apply_patch_permutation else ""
    )
    return (
        f"{model_name}_ps{config.obfuscation.patch_size}"
        f"_gs{config.obfuscation.group_size}{permutation_tag}_s{config.seed}"
    )


def _task_checkpoint_key(config: ExperimentConfig) -> str:
    """Generate a key for task-specific training state."""
    task_cfg = config.task_training
    payload: dict[str, Any] = {
        "name": config.name,
        "model": config.model.hf_model_name_or_path,
        "task": config.model.task,
        "dataset": config.dataset.hf_dataset_name_or_path,
        "subset": config.dataset.subset,
        "train_split": config.dataset.train_split,
        "num_classes": config.model.num_classes,
        "seed": config.seed,
        "task_training": {
            "iterations": task_cfg.iterations if task_cfg else None,
            "learning_rate": task_cfg.learning_rate if task_cfg else None,
            "batch_size": task_cfg.batch_size if task_cfg else None,
        },
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    model_name = config.model.hf_model_name_or_path.replace("/", "_")
    return f"{model_name}_{config.name}_task_{digest}"


def save_checkpoint(
    obfuscator: ChannelWisePatchLevelObfuscator,
    obf_embedding: ObfuscationEmbedding,
    output_dir: str,
    key: str,
) -> str:
    """Save obfuscator and trained embedding to a checkpoint file."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{key}.pt")
    torch.save(
        {
            "obfuscator": obfuscator.state_dict(),
            "obf_embedding": obf_embedding.state_dict(),
        },
        path,
    )
    logger.info(f"Checkpoint saved to {path}")
    return path


def load_checkpoint(
    obfuscator: ChannelWisePatchLevelObfuscator,
    obf_embedding: ObfuscationEmbedding,
    output_dir: str,
    key: str,
) -> bool:
    """Load obfuscator and embedding from checkpoint. Returns True if found."""
    path = os.path.join(output_dir, "checkpoints", f"{key}.pt")
    if not os.path.exists(path):
        return False
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    obfuscator.load_state_dict(ckpt["obfuscator"])
    obf_embedding.load_state_dict(ckpt["obf_embedding"])
    logger.info(f"Checkpoint loaded from {path}")
    return True


def save_task_checkpoint(task, output_dir: str, key: str) -> str:
    """Save task-specific trainable state."""
    ckpt_dir = os.path.join(output_dir, "task_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{key}.pt")
    torch.save(task.task_state_dict(), path)
    logger.info(f"Task checkpoint saved to {path}")
    return path


def load_task_checkpoint(task, output_dir: str, key: str) -> bool:
    """Load task-specific state if it exists."""
    path = os.path.join(output_dir, "task_checkpoints", f"{key}.pt")
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)
    task.load_task_state_dict(state)
    logger.info(f"Task checkpoint loaded from {path}")
    return True


class ExperimentRunner:
    """Orchestrates the full experiment pipeline with checkpoint support."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> dict:
        config = self.config

        logger.info(f"Starting experiment: {config.name}")
        logger.info(
            f"Task: {config.model.task}, Model: {config.model.hf_model_name_or_path}"
        )

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # 1. Load model and processor
        logger.info("Loading model and processor...")
        model = _load_model(config)
        processor = AutoProcessor.from_pretrained(config.model.hf_model_name_or_path)

        # 2. Create adapter
        adapter = ModelAdapter(model)
        vision_config = adapter.get_vision_config()

        # Determine actual image size from processor (more reliable than vision_config)
        _proc_size = getattr(
            getattr(processor, "image_processor", processor), "size", {}
        )
        _proc_h = _proc_w = None
        if _proc_size is not None:
            if isinstance(_proc_size, dict) or hasattr(_proc_size, "get"):
                _proc_h = (
                    _proc_size.get("height")
                    or _proc_size.get("shortest_edge")
                    or getattr(_proc_size, "height", None)
                    or getattr(_proc_size, "shortest_edge", None)
                )
                _proc_w = (
                    _proc_size.get("width")
                    or _proc_size.get("shortest_edge")
                    or getattr(_proc_size, "width", None)
                    or getattr(_proc_size, "shortest_edge", None)
                )
            else:
                _proc_h = getattr(_proc_size, "height", None) or getattr(
                    _proc_size, "shortest_edge", None
                )
                _proc_w = getattr(_proc_size, "width", None) or getattr(
                    _proc_size, "shortest_edge", None
                )

        vc_image_size = getattr(vision_config, "image_size", 224)
        if isinstance(vc_image_size, (list, tuple)):
            vc_image_size = vc_image_size[0]

        # Prefer processor size when available and square
        if _proc_h and _proc_w and _proc_h == _proc_w:
            image_size = _proc_h
        else:
            image_size = vc_image_size

        num_channels = getattr(vision_config, "num_channels", 3)
        model_patch_size = getattr(vision_config, "patch_size", None)
        if model_patch_size is None:
            # Some hierarchical vision configs expose patch_sizes as a list.
            patch_sizes = getattr(vision_config, "patch_sizes", [16])
            model_patch_size = patch_sizes[0]

        embed_dim = getattr(vision_config, "hidden_size", None)
        if embed_dim is None:
            # Some hierarchical vision configs expose hidden_sizes as a list.
            hidden_sizes = getattr(vision_config, "hidden_sizes", [768])
            embed_dim = hidden_sizes[-1]

        logger.info(
            f"Vision config: image_size={image_size}, patch_size={model_patch_size}, embed_dim={embed_dim}"
        )

        # 3. Create obfuscation modules
        obfuscator = ChannelWisePatchLevelObfuscator(
            image_size=image_size,
            num_channels=num_channels,
            patch_size=config.obfuscation.patch_size,
            group_size=config.obfuscation.group_size,
            apply_patch_permutation=config.obfuscation.apply_patch_permutation,
        )

        num_extra_tokens = adapter.get_num_extra_tokens()
        obf_embedding = ObfuscationEmbedding(
            image_size=image_size,
            num_channels=num_channels,
            patch_size=model_patch_size,
            embed_dim=embed_dim,
            num_extra_tokens=num_extra_tokens,
        )

        # 4. Copy frozen params and try loading checkpoint
        adapter.copy_frozen_params(obf_embedding)
        ckpt_key = _checkpoint_key(config)

        if load_checkpoint(obfuscator, obf_embedding, config.output_dir, ckpt_key):
            adapter.copy_frozen_params(obf_embedding)
            save_checkpoint(obfuscator, obf_embedding, config.output_dir, ckpt_key)
            logger.info(
                "Loaded existing checkpoint, refreshed frozen embedding parameters, "
                "and skipped embedding training"
            )
        else:
            # 5. Train deobfuscation embedding
            logger.info("Training obfuscation embedding...")
            trainer = EmbeddingTrainer(
                adapter=adapter,
                obfuscator=obfuscator,
                obf_embedding=obf_embedding,
                processor=processor,
                config=config.embedding_training,
            )
            trainer.train()

            obf_embedding = trainer.obf_embedding
            obfuscator = trainer.obfuscator
            adapter = ModelAdapter(trainer.adapter.model, adapter.spec)

            # Save checkpoint for reuse
            save_checkpoint(obfuscator, obf_embedding, config.output_dir, ckpt_key)

        # 6. Create task
        TaskClass = _import_task_class(config.model.task)
        task = TaskClass(
            adapter=adapter,
            obfuscator=obfuscator,
            obf_embedding=obf_embedding,
            processor=processor,
            config=config,
        )

        # 7. Train task if needed (must happen before evaluation)
        if config.task_training is not None:
            task_ckpt_key = _task_checkpoint_key(config)
            if load_task_checkpoint(task, config.output_dir, task_ckpt_key):
                logger.info("Loaded existing task checkpoint, skipping task training")
            else:
                logger.info("Training task-specific components...")
                task.train_task()
                save_task_checkpoint(task, config.output_dir, task_ckpt_key)

        # 8. Evaluate clean baseline (with trained task head)
        logger.info("Evaluating clean baseline...")
        clean_results = task.evaluate(with_obfuscation=False)
        logger.info(f"Clean results: {clean_results}")

        # 9. Evaluate with obfuscation
        logger.info("Evaluating with obfuscation...")
        obf_results = task.evaluate(with_obfuscation=True)
        logger.info(f"Obfuscated results: {obf_results}")

        # 10. Save results
        results = {
            "experiment": config.name,
            "model": config.model.hf_model_name_or_path,
            "task": config.model.task,
            "clean": dict(clean_results.items()),
            "obfuscated": dict(obf_results.items()),
        }

        os.makedirs(config.output_dir, exist_ok=True)
        output_path = os.path.join(config.output_dir, f"{config.name}_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        write_manifest(
            config=config,
            output_dir=config.output_dir,
            result_file=output_path,
            status="success",
        )
        logger.info(f"Results saved to {output_path}")

        return results


def run_all_experiments(config_dir: str, output_dir: str = "./outputs") -> list[dict]:
    """
    Run all experiment configs in optimal order.

    Groups experiments by base model so that models sharing the same
    HF checkpoint reuse the trained obfuscation embedding checkpoint.
    Orders: same-model experiments run consecutively to benefit from
    cached model weights and shared embedding checkpoints.
    """
    config_files = sorted(Path(config_dir).glob("*.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No YAML configs found in {config_dir}")

    configs = []
    for path in config_files:
        config = ExperimentConfig.from_yaml(str(path))
        config.output_dir = output_dir
        configs.append(config)

    # Group by base model for checkpoint reuse
    model_groups = defaultdict(list)
    for config in configs:
        model_groups[config.model.hf_model_name_or_path].append(config)

    all_results = []
    for model_name, group_configs in model_groups.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Model group: {model_name} ({len(group_configs)} experiments)")
        logger.info(f"{'=' * 60}")

        for config in group_configs:
            try:
                runner = ExperimentRunner(config)
                results = runner.run()
                all_results.append(results)
            except Exception as e:
                logger.error(f"Experiment '{config.name}' failed: {e}", exc_info=True)
                error_path = os.path.join(output_dir, f"{config.name}_results.json")
                os.makedirs(output_dir, exist_ok=True)
                error_result = {
                    "experiment": config.name,
                    "model": config.model.hf_model_name_or_path,
                    "task": config.model.task,
                    "error": str(e),
                }
                with open(error_path, "w") as f:
                    json.dump(error_result, f, indent=2)
                write_manifest(
                    config=config,
                    output_dir=output_dir,
                    result_file=error_path,
                    status="error",
                    error=str(e),
                )
                all_results.append(
                    error_result
                )

    # Save canonical combined results for manuscript tables.
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "all_results.json")
    successful_results = [
        row
        for row in all_results
        if isinstance(row, dict)
        and "error" not in row
        and "clean" in row
        and "obfuscated" in row
    ]
    with open(combined_path, "w") as f:
        json.dump(successful_results, f, indent=2, default=str)
    logger.info(f"\nAll results saved to {combined_path}")

    return all_results
