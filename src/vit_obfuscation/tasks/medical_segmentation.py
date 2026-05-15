from __future__ import annotations

import itertools

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..adapter.model_adapter import ModelAdapter
from ..config.experiment import ExperimentConfig
from ..datasets.loader import load_classification_dataset
from ..embedding.embedding import ObfuscationEmbedding
from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from ..outputs.types import BinarySegmentationEvalOutput
from .base import BaseTask
from .feature_utils import (
    collate_images,
    image_to_binary_mask,
    maybe_limit_dataset,
    output_patch_tokens,
    vision_forward,
)


def _dice_iou_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    preds = preds.bool()
    targets = targets.bool()
    intersection = (preds & targets).sum().float()
    pred_sum = preds.sum().float()
    target_sum = targets.sum().float()
    union = (preds | targets).sum().float()
    dice = (2 * intersection / (pred_sum + target_sum + 1e-8)).item()
    iou = (intersection / (union + 1e-8)).item()
    acc = (preds == targets).float().mean().item()
    return dice, iou, acc


class MedicalBinarySegmentationTask(BaseTask):
    """Binary medical segmentation using frozen ViT patch tokens plus a small head."""

    def __init__(
        self,
        adapter: ModelAdapter,
        obfuscator: ChannelWisePatchLevelObfuscator,
        obf_embedding: ObfuscationEmbedding,
        processor,
        config: ExperimentConfig,
    ) -> None:
        super().__init__(adapter, obfuscator, obf_embedding, processor, config)
        embed_dim = adapter.get_vision_config().hidden_size
        self.segmentation_head = nn.Linear(embed_dim, 1)

    def _mask_column(self) -> str:
        return self.config.dataset.mask_column or self.config.dataset.label_column

    def _features_to_logits(
        self, outputs, output_size: tuple[int, int], head: nn.Module | None = None
    ) -> torch.Tensor:
        head = head or self.segmentation_head
        tokens = output_patch_tokens(outputs)
        B, N, D = tokens.shape
        side = int(N**0.5)
        if side * side != N:
            raise ValueError("Patch-token count must form a square grid for segmentation")
        logits = head(tokens).transpose(1, 2).reshape(B, 1, side, side)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    def forward(self, images, with_obfuscation: bool = False, **kwargs):
        model = self.adapter.model
        inputs = self.process_images(images)
        device = next(model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)
        if with_obfuscation:
            pixel_values = self.obfuscator(pixel_values)
            with self.adapter.obfuscation_mode(self.obf_embedding):
                return vision_forward(model, pixel_values)
        return vision_forward(model, pixel_values)

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
            subset=cfg.dataset.subset,
        )
        train_dataset = maybe_limit_dataset(train_dataset, cfg.evaluation.max_samples)
        mask_column = self._mask_column()

        def collate_fn(batch):
            return {
                cfg.dataset.input_column: collate_images(batch, cfg.dataset.input_column),
                mask_column: [item[mask_column] for item in batch],
            }

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=task_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        accelerator = accelerate.Accelerator()
        model = self.adapter.model
        head = self.segmentation_head
        for param in model.parameters():
            param.requires_grad_(False)
        optimizer = torch.optim.AdamW(head.parameters(), lr=task_cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=task_cfg.learning_rate,
            total_steps=task_cfg.iterations,
        )
        model, head, optimizer, scheduler, dataloader = accelerator.prepare(
            model, head, optimizer, scheduler, dataloader
        )
        model.eval()
        head.train()

        pbar = tqdm(
            itertools.islice(itertools.cycle(dataloader), task_cfg.iterations),
            total=task_cfg.iterations,
            desc="Training medical segmentation head",
        )
        for batch in pbar:
            optimizer.zero_grad()
            images = batch[cfg.dataset.input_column]
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(accelerator.device)
            masks = torch.stack(
                [image_to_binary_mask(mask) for mask in batch[mask_column]]
            ).to(accelerator.device)
            with torch.no_grad():
                outputs = vision_forward(model, pixel_values)
            logits = self._features_to_logits(outputs, masks.shape[-2:], head=head)
            loss = F.binary_cross_entropy_with_logits(logits, masks)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        self.adapter.model = accelerator.unwrap_model(model).cpu()
        self.segmentation_head = accelerator.unwrap_model(head).cpu()

    def evaluate(self, with_obfuscation: bool = False) -> BinarySegmentationEvalOutput:
        cfg = self.config
        _, eval_dataset = load_classification_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
            subset=cfg.dataset.subset,
        )
        eval_dataset = maybe_limit_dataset(eval_dataset, cfg.evaluation.max_samples)
        mask_column = self._mask_column()

        def collate_fn(batch):
            return {
                cfg.dataset.input_column: collate_images(batch, cfg.dataset.input_column),
                mask_column: [item[mask_column] for item in batch],
            }

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        accelerator = accelerate.Accelerator()
        model = self.adapter.model
        head = self.segmentation_head
        model, head, dataloader = accelerator.prepare(model, head, dataloader)
        if with_obfuscation:
            self.obfuscation_modules_to(accelerator.device)

        dice_values = []
        iou_values = []
        acc_values = []
        model.eval()
        head.eval()
        for batch in tqdm(dataloader, desc="Evaluating medical segmentation"):
            with torch.no_grad():
                images = batch[cfg.dataset.input_column]
                inputs = self.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(accelerator.device)
                masks = torch.stack(
                    [image_to_binary_mask(mask) for mask in batch[mask_column]]
                ).to(accelerator.device)
                if with_obfuscation:
                    pixel_values = self.obfuscator(pixel_values)
                    unwrapped = accelerator.unwrap_model(model)
                    adapter = ModelAdapter(unwrapped, self.adapter.spec)
                    adapter.swap_to_obfuscation(self.obf_embedding)
                    outputs = vision_forward(model, pixel_values)
                    adapter.swap_to_original()
                else:
                    outputs = vision_forward(model, pixel_values)
                logits = self._features_to_logits(outputs, masks.shape[-2:], head=head)
                preds = torch.sigmoid(logits) >= 0.5
                dice, iou, acc = _dice_iou_accuracy(preds.cpu(), masks.cpu() >= 0.5)
                dice_values.append(dice)
                iou_values.append(iou)
                acc_values.append(acc)

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")
        self.adapter.model = accelerator.unwrap_model(model).cpu()
        self.segmentation_head = accelerator.unwrap_model(head).cpu()

        return BinarySegmentationEvalOutput(
            dice=sum(dice_values) / len(dice_values) if dice_values else 0.0,
            iou=sum(iou_values) / len(iou_values) if iou_values else 0.0,
            pixel_accuracy=sum(acc_values) / len(acc_values) if acc_values else 0.0,
        )
