from __future__ import annotations

import itertools
from typing import Any

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..adapter.model_adapter import ModelAdapter
from ..config.experiment import ExperimentConfig
from ..datasets.loader import load_classification_dataset
from ..embedding.embedding import ObfuscationEmbedding
from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from ..outputs.types import ModelOutputWithObfuscation, SegmentationEvalOutput
from .base import BaseTask


def _seg_collate_fn(batch, input_column, label_column):
    """Collate that keeps images and label masks as lists."""
    images = [item[input_column] for item in batch]
    labels = [item[label_column] for item in batch]
    return {input_column: images, label_column: labels}


def compute_miou(
    pred_masks: np.ndarray, gt_masks: np.ndarray, num_classes: int
) -> dict:
    """Compute mean IoU and related metrics."""
    iou_per_class = {}
    acc_per_class = {}
    total_correct = 0
    total_pixels = 0

    for cls_id in range(num_classes):
        pred_cls = pred_masks == cls_id
        gt_cls = gt_masks == cls_id

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union > 0:
            iou_per_class[cls_id] = intersection / union
        if gt_cls.sum() > 0:
            acc_per_class[cls_id] = intersection / gt_cls.sum()

        total_correct += intersection
        total_pixels += gt_cls.sum()

    mean_iou = np.mean(list(iou_per_class.values())) if iou_per_class else 0.0
    mean_acc = np.mean(list(acc_per_class.values())) if acc_per_class else 0.0
    overall_acc = total_correct / total_pixels if total_pixels > 0 else 0.0

    return {
        "mean_iou": float(mean_iou),
        "mean_accuracy": float(mean_acc),
        "overall_accuracy": float(overall_acc),
    }


class SegmentationTask(BaseTask):
    """Semantic segmentation (e.g., SegFormer)."""

    def __init__(
        self,
        adapter: ModelAdapter,
        obfuscator: ChannelWisePatchLevelObfuscator,
        obf_embedding: ObfuscationEmbedding,
        processor,
        config: ExperimentConfig,
    ) -> None:
        super().__init__(adapter, obfuscator, obf_embedding, processor, config)

    def forward(self, images, with_obfuscation: bool = False, **kwargs):
        model = self.adapter.model
        inputs = self.process_images(images)
        device = next(model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)

        if with_obfuscation:
            obfuscated = self.obfuscator(pixel_values)
            self.adapter.swap_to_obfuscation(self.obf_embedding)
            outputs = model(pixel_values=obfuscated)
            self.adapter.swap_to_original()
        else:
            outputs = model(pixel_values=pixel_values)

        return ModelOutputWithObfuscation(
            obfuscated_images=pixel_values if with_obfuscation else None,
            model_outputs=outputs,
        )

    def train_task(self) -> None:
        cfg = self.config
        task_cfg = cfg.task_training
        if task_cfg is None:
            return

        train_dataset, _ = load_classification_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
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

        collate = lambda batch: _seg_collate_fn(
            batch, cfg.dataset.input_column, cfg.dataset.label_column
        )
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=task_cfg.batch_size,
            shuffle=True,
            collate_fn=collate,
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
            desc="Training segmentation",
        )

        for batch in pbar:
            optimizer.zero_grad()

            images = batch[cfg.dataset.input_column]
            labels = torch.stack(
                [torch.as_tensor(np.array(l)) for l in batch[cfg.dataset.label_column]]
            ).to(accelerator.device)

            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(accelerator.device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.adapter.model = accelerator.unwrap_model(model).cpu()

    def evaluate(self, with_obfuscation: bool = False) -> SegmentationEvalOutput:
        cfg = self.config

        _, eval_dataset = load_classification_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
        )

        num_classes = cfg.dataset.num_classes or cfg.model.num_classes or 150

        accelerator = accelerate.Accelerator()

        collate = lambda batch: _seg_collate_fn(
            batch, cfg.dataset.input_column, cfg.dataset.label_column
        )
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate,
        )

        model = self.adapter.model
        model, dataloader = accelerator.prepare(model, dataloader)
        unwrapped = accelerator.unwrap_model(model)

        if with_obfuscation:
            self.obfuscation_modules_to(accelerator.device)

        all_preds = []
        all_labels = []

        model.eval()
        for batch in tqdm(dataloader, desc="Evaluating segmentation"):
            with torch.no_grad():
                images = batch[cfg.dataset.input_column]
                labels = torch.stack(
                    [
                        torch.as_tensor(np.array(l))
                        for l in batch[cfg.dataset.label_column]
                    ]
                )

                inputs = self.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(accelerator.device)

                if with_obfuscation:
                    pixel_values = self.obfuscator(pixel_values)
                    adapter = ModelAdapter(unwrapped, self.adapter.spec)
                    adapter.swap_to_obfuscation(self.obf_embedding)

                outputs = model(pixel_values=pixel_values)

                if with_obfuscation:
                    adapter.swap_to_original()

                logits = outputs.logits
                upsampled = F.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                preds = upsampled.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = compute_miou(all_preds, all_labels, num_classes)
        return SegmentationEvalOutput(**metrics)


class ZeroShotSegmentationTask(BaseTask):
    """Zero-shot segmentation (e.g., CLIPSeg)."""

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

    def evaluate(self, with_obfuscation: bool = False) -> SegmentationEvalOutput:
        cfg = self.config

        _, eval_dataset = load_classification_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
        )

        id2label = cfg.dataset.id2label or cfg.model.id2label
        if id2label is None:
            raise ValueError("id2label mapping required for zero-shot segmentation")
        label_names = [name for _, name in sorted(id2label.items())]
        num_classes = len(label_names)

        accelerator = accelerate.Accelerator()

        collate = lambda batch: _seg_collate_fn(
            batch, cfg.dataset.input_column, cfg.dataset.label_column
        )
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate,
        )

        model = self.adapter.model
        model, dataloader = accelerator.prepare(model, dataloader)
        unwrapped = accelerator.unwrap_model(model)

        if with_obfuscation:
            self.obfuscation_modules_to(accelerator.device)

        all_preds = []
        all_labels = []

        model.eval()
        for batch in tqdm(dataloader, desc="Evaluating zero-shot segmentation"):
            with torch.no_grad():
                images = batch[cfg.dataset.input_column]
                labels = torch.stack(
                    [
                        torch.as_tensor(np.array(l))
                        for l in batch[cfg.dataset.label_column]
                    ]
                )

                inputs = self.processor(
                    images=images,
                    text=[label_names] * len(images),
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {
                    k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                if with_obfuscation:
                    obfuscated = self.obfuscator(inputs["pixel_values"])
                    adapter = ModelAdapter(unwrapped, self.adapter.spec)
                    adapter.swap_to_obfuscation(self.obf_embedding)
                    inputs["pixel_values"] = obfuscated

                outputs = model(**inputs)

                if with_obfuscation:
                    adapter.swap_to_original()

                # CLIPSeg outputs logits per text query
                logits = outputs.logits
                if logits.dim() == 4:
                    upsampled = F.interpolate(
                        logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    preds = upsampled.argmax(dim=1)
                else:
                    preds = (logits > 0).long()

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = compute_miou(all_preds, all_labels, num_classes)
        return SegmentationEvalOutput(**metrics)
