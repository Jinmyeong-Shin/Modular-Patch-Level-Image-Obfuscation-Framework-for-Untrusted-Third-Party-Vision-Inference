from __future__ import annotations

import itertools
import json
from typing import Any

import accelerate
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

from transformers import AutoModelForObjectDetection, OwlViTForObjectDetection

from ..adapter.model_adapter import ModelAdapter
from ..config.experiment import ExperimentConfig
from ..datasets.loader import load_detection_dataset
from ..embedding.embedding import ObfuscationEmbedding
from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from ..outputs.types import DetectionEvalOutput, ModelOutputWithObfuscation
from .base import BaseTask


class ObjectDetectionTask(BaseTask):
    """Fine-tunable object detection (e.g., YOLOS)."""

    def __init__(
        self,
        adapter: ModelAdapter,
        obfuscator: ChannelWisePatchLevelObfuscator,
        obf_embedding: ObfuscationEmbedding,
        processor,
        config: ExperimentConfig,
    ) -> None:
        super().__init__(adapter, obfuscator, obf_embedding, processor, config)

    def _get_processor_kwargs(self) -> dict:
        """Get processor kwargs for detection (e.g., image size)."""
        return {}

    def forward(self, images, labels=None, with_obfuscation: bool = False, **kwargs):
        model = self.adapter.model
        proc_kwargs = self._get_processor_kwargs()

        if labels is not None:
            inputs = self.processor(
                images=images,
                annotations=labels,
                return_tensors="pt",
                **proc_kwargs,
            )
        else:
            inputs = self.processor(images=images, return_tensors="pt", **proc_kwargs)

        device = next(model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        if with_obfuscation:
            obfuscated = self.obfuscator(inputs["pixel_values"])
            self.adapter.swap_to_obfuscation(self.obf_embedding)
            inputs["pixel_values"] = obfuscated
            outputs = model(**inputs)
            self.adapter.swap_to_original()
        else:
            outputs = model(**inputs)

        return ModelOutputWithObfuscation(
            obfuscated_images=inputs.get("pixel_values") if with_obfuscation else None,
            model_outputs=outputs,
        )

    def train_task(self) -> None:
        cfg = self.config
        task_cfg = cfg.task_training
        if task_cfg is None:
            return

        train_dataset, _ = load_detection_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.label_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
            subset=cfg.dataset.subset,
        )

        accelerator = accelerate.Accelerator()

        optimizer = torch.optim.AdamW(
            self.adapter.model.parameters(), lr=task_cfg.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=task_cfg.learning_rate,
            total_steps=task_cfg.iterations,
        )

        def collate_fn(batch):
            processed_labels = []
            for item in batch:
                label_data = item[cfg.dataset.label_column]
                if isinstance(label_data, str):
                    label_data = json.loads(label_data)
                processed_labels.append(label_data)
            return {
                cfg.dataset.input_column: [
                    item[cfg.dataset.input_column] for item in batch
                ],
                cfg.dataset.label_column: processed_labels,
            }

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=task_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        model = self.adapter.model
        model, optimizer, scheduler, dataloader = accelerator.prepare(
            model,
            optimizer,
            scheduler,
            dataloader,
        )

        model.train()
        pbar = tqdm(
            itertools.islice(itertools.cycle(dataloader), task_cfg.iterations),
            total=task_cfg.iterations,
            desc="Training detector",
        )

        for batch in pbar:
            optimizer.zero_grad()

            outputs = self.forward(
                images=batch[cfg.dataset.input_column],
                labels=batch[cfg.dataset.label_column],
            )

            loss = outputs.model_outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.adapter.model = accelerator.unwrap_model(model).cpu()

    def evaluate(self, with_obfuscation: bool = False) -> DetectionEvalOutput:
        cfg = self.config

        _, eval_dataset = load_detection_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.label_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
            subset=cfg.dataset.subset,
        )

        def collate_fn(batch):
            processed_labels = []
            for item in batch:
                label_data = item[cfg.dataset.label_column]
                if isinstance(label_data, str):
                    label_data = json.loads(label_data)
                processed_labels.append(label_data)
            images = [item[cfg.dataset.input_column] for item in batch]
            target_sizes = torch.tensor([img.size[::-1] for img in images])
            return images, processed_labels, target_sizes

        accelerator = accelerate.Accelerator()

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = self.adapter.model
        model, dataloader = accelerator.prepare(model, dataloader)

        if with_obfuscation:
            self.obfuscation_modules_to(accelerator.device)

        unwrapped_model = accelerator.unwrap_model(model)
        metric = MeanAveragePrecision(
            box_format="xyxy",
            max_detection_thresholds=[1, 10, 100],
        ).to(accelerator.device)

        model.eval()
        for images, targets, target_sizes in tqdm(
            dataloader, desc="Evaluating detector"
        ):
            with torch.no_grad():
                proc_kwargs = self._get_processor_kwargs()
                inputs = self.processor(
                    images=images, return_tensors="pt", **proc_kwargs
                )
                pixel_values = inputs["pixel_values"].to(accelerator.device)

                if with_obfuscation:
                    pixel_values = self.obfuscator(pixel_values)
                    adapter = ModelAdapter(unwrapped_model, self.adapter.spec)
                    adapter.swap_to_obfuscation(self.obf_embedding)

                outputs = model(pixel_values=pixel_values)

                if with_obfuscation:
                    adapter.swap_to_original()

                orig_target_sizes = target_sizes.to(accelerator.device)
                results = self.processor.post_process_object_detection(
                    outputs,
                    target_sizes=orig_target_sizes,
                    threshold=0.1,
                )

                preds = [
                    {"boxes": r["boxes"], "scores": r["scores"], "labels": r["labels"]}
                    for r in results
                ]

                formatted_targets = []
                for target_annos in targets:
                    boxes = []
                    labels = []
                    for anno in target_annos:
                        box = anno["bbox"]
                        boxes.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                        labels.append(anno["category_id"])
                    formatted_targets.append(
                        {
                            "boxes": torch.tensor(boxes, dtype=torch.float32).to(
                                accelerator.device
                            ),
                            "labels": torch.tensor(labels, dtype=torch.int64).to(
                                accelerator.device
                            ),
                        }
                    )
                metric.update(preds, formatted_targets)

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")

        computed = metric.compute()
        scalar_metrics = {
            k: v.item()
            for k, v in computed.items()
            if v.numel() == 1 and not k.endswith("_per_class") and k != "classes"
        }
        return DetectionEvalOutput(**scalar_metrics)


class ZeroShotObjectDetectionTask(BaseTask):
    """Zero-shot object detection (e.g., OWL-ViT)."""

    def __init__(
        self,
        adapter: ModelAdapter,
        obfuscator: ChannelWisePatchLevelObfuscator,
        obf_embedding: ObfuscationEmbedding,
        processor,
        config: ExperimentConfig,
    ) -> None:
        super().__init__(adapter, obfuscator, obf_embedding, processor, config)

    def forward(self, images, text=None, with_obfuscation: bool = False, **kwargs):
        model = self.adapter.model
        inputs = self.processor(
            images=images,
            text=text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        if with_obfuscation:
            obfuscated = self.obfuscator(inputs["pixel_values"])
            self.adapter.swap_to_obfuscation(self.obf_embedding)
            inputs["pixel_values"] = obfuscated
            outputs = model(**inputs)
            self.adapter.swap_to_original()
        else:
            outputs = model(**inputs)

        return ModelOutputWithObfuscation(
            obfuscated_images=inputs.get("pixel_values") if with_obfuscation else None,
            model_outputs=outputs,
        )

    def train_task(self) -> None:
        pass  # Zero-shot: no task training

    def evaluate(self, with_obfuscation: bool = False) -> DetectionEvalOutput:
        cfg = self.config

        _, eval_dataset = load_detection_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.label_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
            subset=cfg.dataset.subset,
        )

        id2label = cfg.dataset.id2label or cfg.model.id2label
        if id2label is None:
            raise ValueError("id2label mapping required for zero-shot detection")
        label_names = [name for _, name in sorted(id2label.items())]

        def collate_fn(batch):
            processed_labels = []
            for item in batch:
                label_data = item[cfg.dataset.label_column]
                if isinstance(label_data, str):
                    label_data = json.loads(label_data)
                processed_labels.append(label_data)
            images = [item[cfg.dataset.input_column] for item in batch]
            target_sizes = torch.tensor([img.size[::-1] for img in images])
            return images, processed_labels, target_sizes

        accelerator = accelerate.Accelerator()

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = self.adapter.model
        model, dataloader = accelerator.prepare(model, dataloader)

        if with_obfuscation:
            self.obfuscation_modules_to(accelerator.device)

        unwrapped_model = accelerator.unwrap_model(model)
        metric = MeanAveragePrecision(
            box_format="xyxy",
            max_detection_thresholds=[1, 10, 100],
        ).to(accelerator.device)

        model.eval()
        for images, targets, target_sizes in tqdm(
            dataloader, desc="Evaluating zero-shot detector"
        ):
            with torch.no_grad():
                text_queries = [label_names] * len(images)

                inputs = self.processor(
                    images=images,
                    text=text_queries,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {
                    k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                if with_obfuscation:
                    obfuscated = self.obfuscator(inputs["pixel_values"])
                    adapter = ModelAdapter(unwrapped_model, self.adapter.spec)
                    adapter.swap_to_obfuscation(self.obf_embedding)
                    inputs["pixel_values"] = obfuscated

                outputs = model(**inputs)

                if with_obfuscation:
                    adapter.swap_to_original()

                orig_target_sizes = target_sizes.to(accelerator.device)
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    target_sizes=orig_target_sizes,
                    threshold=0.1,
                )

                preds = [
                    {"boxes": r["boxes"], "scores": r["scores"], "labels": r["labels"]}
                    for r in results
                ]

                formatted_targets = []
                for target_annos in targets:
                    boxes = []
                    labels = []
                    for anno in target_annos:
                        box = anno["bbox"]
                        boxes.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                        labels.append(anno["category_id"])
                    formatted_targets.append(
                        {
                            "boxes": torch.tensor(boxes, dtype=torch.float32).to(
                                accelerator.device
                            ),
                            "labels": torch.tensor(labels, dtype=torch.int64).to(
                                accelerator.device
                            ),
                        }
                    )
                metric.update(preds, formatted_targets)

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")

        computed = metric.compute()
        scalar_metrics = {
            k: v.item()
            for k, v in computed.items()
            if v.numel() == 1 and not k.endswith("_per_class") and k != "classes"
        }
        return DetectionEvalOutput(**scalar_metrics)
