from __future__ import annotations

from typing import Any

import datasets
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm

from ..adapter.model_adapter import ModelAdapter
from ..config.experiment import ExperimentConfig
from ..datasets.loader import load_classification_dataset
from ..embedding.embedding import ObfuscationEmbedding
from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from ..outputs.types import AnomalyEvalOutput
from .base import BaseTask
from .feature_utils import (
    collate_images,
    image_to_binary_mask,
    maybe_limit_dataset,
    normalize_features,
    output_patch_tokens,
    pooled_features_from_outputs,
    vision_forward,
)


def _is_anomaly(label, normal_label=None) -> int:
    if label is None:
        return 0
    if isinstance(label, torch.Tensor):
        label = label.item()
    if normal_label is not None:
        return int(str(label).lower() != str(normal_label).lower())
    if isinstance(label, str):
        return int(label.lower() not in {"good", "normal", "ok", "0"})
    return int(label != 0)


def _safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(average_precision_score(labels, scores))


def _approximate_pro(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    labels_np = labels.flatten().cpu().numpy()
    scores_np = scores.flatten().cpu().numpy()
    if len(set(labels_np.tolist())) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels_np, scores_np)
    keep = fpr <= 0.3
    if not keep.any():
        return None
    return float(tpr[keep].mean())


def _as_rgb(image: Any) -> Any:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return image


def _zero_mask_like(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return Image.new("L", image.size, 0)
    return Image.new("L", (1, 1), 0)


def _as_mask(mask: Any) -> Any:
    if isinstance(mask, Image.Image):
        return mask.convert("L")
    return mask


def _load_hf_dataset(config: ExperimentConfig, split: str):
    resolved_name = config.dataset.hf_dataset_name_or_path
    if config.dataset.subset is not None:
        return datasets.load_dataset(
            resolved_name,
            name=config.dataset.subset,
            split=split,
        )
    return datasets.load_dataset(resolved_name, split=split)


def _load_guided_anomaly_dataset(
    config: ExperimentConfig,
    split: str,
    max_samples: int | None,
    *,
    train_normals_only: bool,
) -> list[dict]:
    """Build a bounded anomaly dataset from anomaly images plus guide normals.

    Some compact MVTec-style Hugging Face datasets expose an anomalous image, a
    normal guide image, and a mask per row. This materializes only the requested
    split slice and turns each guide into a normal sample, avoiding the full
    Voxel51 MVTec load that can block artifact generation.
    """

    normal_column = config.dataset.normal_column
    if normal_column is None:
        raise ValueError("normal_column is required for guided anomaly loading")

    input_column = config.dataset.input_column
    label_column = config.dataset.label_column
    mask_column = config.dataset.mask_column
    normal_label = config.dataset.normal_label if config.dataset.normal_label is not None else 0

    dataset = _load_hf_dataset(config, split)
    records: list[dict] = []
    for item in dataset:
        normal_image = item.get(normal_column)
        anomaly_image = item.get(input_column)
        if normal_image is None or anomaly_image is None:
            continue

        normal_record = {
            input_column: _as_rgb(normal_image),
            label_column: normal_label,
        }
        if mask_column is not None:
            normal_record[mask_column] = _zero_mask_like(normal_image)
        records.append(normal_record)

        if not train_normals_only:
            anomaly_record = {
                input_column: _as_rgb(anomaly_image),
                label_column: 1,
            }
            if mask_column is not None:
                anomaly_record[mask_column] = _as_mask(item.get(mask_column))
            records.append(anomaly_record)

        if max_samples is not None and len(records) >= max_samples:
            break

    if max_samples is not None:
        records = records[:max_samples]
    if not records:
        raise ValueError(f"No anomaly samples loaded from {config.dataset.hf_dataset_name_or_path}")
    return records


class AnomalyDetectionTask(BaseTask):
    """Frozen-feature industrial anomaly detection and optional localization."""

    def __init__(
        self,
        adapter: ModelAdapter,
        obfuscator: ChannelWisePatchLevelObfuscator,
        obf_embedding: ObfuscationEmbedding,
        processor,
        config: ExperimentConfig,
    ) -> None:
        super().__init__(adapter, obfuscator, obf_embedding, processor, config)
        self.normal_image_bank: torch.Tensor | None = None
        self.normal_patch_bank: torch.Tensor | None = None

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
        if cfg.dataset.normal_column:
            train_dataset = _load_guided_anomaly_dataset(
                cfg,
                cfg.dataset.train_split,
                cfg.evaluation.max_samples,
                train_normals_only=True,
            )
        else:
            train_dataset, _ = load_classification_dataset(
                cfg.dataset.hf_dataset_name_or_path,
                cfg.dataset.input_column,
                cfg.dataset.train_split,
                cfg.dataset.eval_split,
                subset=cfg.dataset.subset,
            )
            train_dataset = maybe_limit_dataset(train_dataset, cfg.evaluation.max_samples)

        def collate_fn(batch):
            return {
                cfg.dataset.input_column: collate_images(batch, cfg.dataset.input_column),
                cfg.dataset.label_column: [
                    item.get(cfg.dataset.label_column) for item in batch
                ],
            }

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.adapter.model.to(device).eval()
        self.obfuscation_modules_to(device)

        image_features = []
        patch_features = []
        for batch in tqdm(dataloader, desc="Building normal feature bank"):
            normal_indices = [
                i
                for i, label in enumerate(batch[cfg.dataset.label_column])
                if _is_anomaly(label, cfg.dataset.normal_label) == 0
            ]
            if not normal_indices:
                continue
            images = [batch[cfg.dataset.input_column][i] for i in normal_indices]
            with torch.no_grad():
                inputs = self.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                outputs = vision_forward(model, pixel_values)
                image_features.append(
                    normalize_features(pooled_features_from_outputs(outputs)).cpu()
                )
                try:
                    patches = output_patch_tokens(outputs)
                    patch_features.append(normalize_features(patches.reshape(-1, patches.shape[-1])).cpu())
                except ValueError:
                    pass

        if not image_features:
            raise ValueError("No normal training samples found for anomaly detection")

        self.normal_image_bank = torch.cat(image_features, dim=0)
        self.normal_patch_bank = torch.cat(patch_features, dim=0) if patch_features else None
        self.adapter.model = model.cpu()
        self.obfuscation_modules_to("cpu")

    def evaluate(self, with_obfuscation: bool = False) -> AnomalyEvalOutput:
        cfg = self.config
        if self.normal_image_bank is None:
            self.train_task()

        if cfg.dataset.normal_column:
            eval_dataset = _load_guided_anomaly_dataset(
                cfg,
                cfg.dataset.eval_split,
                cfg.evaluation.max_samples,
                train_normals_only=False,
            )
        else:
            _, eval_dataset = load_classification_dataset(
                cfg.dataset.hf_dataset_name_or_path,
                cfg.dataset.input_column,
                cfg.dataset.train_split,
                cfg.dataset.eval_split,
                subset=cfg.dataset.subset,
            )
            eval_dataset = maybe_limit_dataset(eval_dataset, cfg.evaluation.max_samples)

        def collate_fn(batch):
            result = {
                cfg.dataset.input_column: collate_images(batch, cfg.dataset.input_column),
                cfg.dataset.label_column: [
                    item.get(cfg.dataset.label_column) for item in batch
                ],
            }
            if cfg.dataset.mask_column:
                result[cfg.dataset.mask_column] = [
                    item.get(cfg.dataset.mask_column) for item in batch
                ]
            return result

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.adapter.model.to(device).eval()
        normal_image_bank = self.normal_image_bank.to(device)
        normal_patch_bank = (
            self.normal_patch_bank.to(device) if self.normal_patch_bank is not None else None
        )
        if with_obfuscation:
            self.obfuscation_modules_to(device)

        image_labels: list[int] = []
        image_scores: list[float] = []
        pixel_labels = []
        pixel_scores = []

        for batch in tqdm(dataloader, desc="Evaluating anomaly detection"):
            with torch.no_grad():
                images = batch[cfg.dataset.input_column]
                labels = [
                    _is_anomaly(label, cfg.dataset.normal_label)
                    for label in batch[cfg.dataset.label_column]
                ]
                inputs = self.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)

                if with_obfuscation:
                    pixel_values = self.obfuscator(pixel_values)
                    adapter = ModelAdapter(model, self.adapter.spec)
                    adapter.swap_to_obfuscation(self.obf_embedding)
                    outputs = vision_forward(model, pixel_values)
                    adapter.swap_to_original()
                else:
                    outputs = vision_forward(model, pixel_values)

                features = normalize_features(pooled_features_from_outputs(outputs))
                distances = torch.cdist(features, normal_image_bank).min(dim=1).values
                image_labels.extend(labels)
                image_scores.extend(distances.detach().cpu().tolist())

                if cfg.dataset.mask_column and normal_patch_bank is not None:
                    patches = normalize_features(output_patch_tokens(outputs))
                    B, N, D = patches.shape
                    side = int(N**0.5)
                    if side * side == N:
                        patch_scores = torch.cdist(
                            patches.reshape(-1, D), normal_patch_bank
                        ).min(dim=1).values.view(B, 1, side, side)
                        masks = torch.stack(
                            [
                                image_to_binary_mask(mask)
                                for mask in batch[cfg.dataset.mask_column]
                            ]
                        ).to(device)
                        upsampled_scores = F.interpolate(
                            patch_scores,
                            size=masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        pixel_labels.append(masks.detach().cpu())
                        pixel_scores.append(upsampled_scores.detach().cpu())

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")
        self.adapter.model = model.cpu()

        pixel_auc = None
        pro_score = None
        if pixel_labels and pixel_scores:
            labels_tensor = torch.cat(pixel_labels, dim=0)
            scores_tensor = torch.cat(pixel_scores, dim=0)
            pixel_auc = _safe_auc(
                labels_tensor.flatten().int().tolist(),
                scores_tensor.flatten().tolist(),
            )
            pro_score = _approximate_pro(labels_tensor, scores_tensor)

        return AnomalyEvalOutput(
            image_auroc=_safe_auc(image_labels, image_scores),
            image_average_precision=_safe_ap(image_labels, image_scores),
            pixel_auroc=pixel_auc,
            pro_score=pro_score,
        )
