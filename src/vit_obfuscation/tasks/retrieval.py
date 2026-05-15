from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..adapter.model_adapter import ModelAdapter
from ..config.experiment import ExperimentConfig
from ..datasets.loader import load_classification_dataset
from ..embedding.embedding import ObfuscationEmbedding
from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from ..outputs.types import RetrievalEvalOutput
from .base import BaseTask
from .feature_utils import (
    collate_images,
    first_text,
    labels_to_tensor,
    maybe_limit_dataset,
    normalize_features,
    pooled_features_from_outputs,
    vision_forward,
)


def _recall_at_k(similarity: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    hits = 0
    valid = 0
    for idx in range(similarity.shape[0]):
        scores = similarity[idx].clone()
        scores[idx] = -float("inf")
        relevant = (labels == labels[idx]).nonzero(as_tuple=False).flatten()
        relevant = relevant[relevant != idx]
        if relevant.numel() == 0:
            continue
        valid += 1
        topk = scores.topk(min(k, scores.numel())).indices
        hits += int(torch.isin(topk, relevant).any().item())
    return hits / valid if valid else 0.0


def _mean_average_precision(similarity: torch.Tensor, labels: torch.Tensor) -> float:
    ap_values = []
    for idx in range(similarity.shape[0]):
        scores = similarity[idx].clone()
        scores[idx] = -float("inf")
        order = scores.argsort(descending=True)
        relevant = labels[order] == labels[idx]
        relevant[order == idx] = False
        total_relevant = relevant.sum().item()
        if total_relevant == 0:
            continue
        cumulative = relevant.float().cumsum(dim=0)
        precision = cumulative / torch.arange(1, len(relevant) + 1, device=scores.device)
        ap_values.append((precision * relevant.float()).sum().item() / total_relevant)
    return float(sum(ap_values) / len(ap_values)) if ap_values else 0.0


def _paired_recall(similarity: torch.Tensor, k: int, dim: int) -> float:
    ranks = similarity.argsort(dim=dim, descending=True)
    hits = 0
    total = similarity.shape[0]
    for idx in range(total):
        if dim == 1:
            topk = ranks[idx, : min(k, ranks.shape[1])]
        else:
            topk = ranks[: min(k, ranks.shape[0]), idx]
        hits += int((topk == idx).any().item())
    return hits / total if total else 0.0


class ImageRetrievalTask(BaseTask):
    """Image-image retrieval using frozen visual features and label relevance."""

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
        pass

    def evaluate(self, with_obfuscation: bool = False) -> RetrievalEvalOutput:
        cfg = self.config
        _, eval_dataset = load_classification_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
            subset=cfg.dataset.subset,
        )
        eval_dataset = maybe_limit_dataset(eval_dataset, cfg.evaluation.max_samples)

        def collate_fn(batch):
            return {
                cfg.dataset.input_column: collate_images(batch, cfg.dataset.input_column),
                cfg.dataset.label_column: [item[cfg.dataset.label_column] for item in batch],
            }

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.adapter.model.to(device).eval()
        if with_obfuscation:
            self.obfuscation_modules_to(device)

        all_features = []
        all_labels = []
        for batch in tqdm(dataloader, desc="Evaluating image retrieval"):
            with torch.no_grad():
                images = batch[cfg.dataset.input_column]
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
                all_features.append(
                    normalize_features(pooled_features_from_outputs(outputs)).cpu()
                )
                all_labels.extend(batch[cfg.dataset.label_column])

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")
        self.adapter.model = model.cpu()

        features = torch.cat(all_features, dim=0)
        labels = labels_to_tensor(all_labels)
        similarity = features @ features.T
        return RetrievalEvalOutput(
            recall_at_1=_recall_at_k(similarity, labels, 1),
            recall_at_5=_recall_at_k(similarity, labels, 5),
            recall_at_10=_recall_at_k(similarity, labels, 10),
            map=_mean_average_precision(similarity, labels),
        )


class ImageTextRetrievalTask(BaseTask):
    """CLIP-style paired image-text retrieval."""

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
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if with_obfuscation:
            obfuscated = self.obfuscator(inputs["pixel_values"])
            with self.adapter.obfuscation_mode(self.obf_embedding):
                inputs["pixel_values"] = obfuscated
                return model(**inputs)
        return model(**inputs)

    def train_task(self) -> None:
        pass

    def evaluate(self, with_obfuscation: bool = False) -> RetrievalEvalOutput:
        cfg = self.config
        text_column = cfg.dataset.text_column or cfg.dataset.label_column
        _, eval_dataset = load_classification_dataset(
            cfg.dataset.hf_dataset_name_or_path,
            cfg.dataset.input_column,
            cfg.dataset.train_split,
            cfg.dataset.eval_split,
            subset=cfg.dataset.subset,
        )
        eval_dataset = maybe_limit_dataset(eval_dataset, cfg.evaluation.max_samples)

        def collate_fn(batch):
            return {
                cfg.dataset.input_column: collate_images(batch, cfg.dataset.input_column),
                text_column: [first_text(item[text_column]) for item in batch],
            }

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.adapter.model.to(device).eval()
        if with_obfuscation:
            self.obfuscation_modules_to(device)

        image_features = []
        text_features = []
        for batch in tqdm(dataloader, desc="Evaluating image-text retrieval"):
            with torch.no_grad():
                inputs = self.processor(
                    images=batch[cfg.dataset.input_column],
                    text=batch[text_column],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
                if with_obfuscation:
                    inputs["pixel_values"] = self.obfuscator(inputs["pixel_values"])
                    adapter = ModelAdapter(model, self.adapter.spec)
                    adapter.swap_to_obfuscation(self.obf_embedding)
                    img = model.get_image_features(pixel_values=inputs["pixel_values"])
                    adapter.swap_to_original()
                else:
                    img = model.get_image_features(pixel_values=inputs["pixel_values"])
                txt = model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                image_features.append(normalize_features(img).cpu())
                text_features.append(normalize_features(txt).cpu())

        if with_obfuscation:
            self.obfuscation_modules_to("cpu")
        self.adapter.model = model.cpu()

        image_matrix = torch.cat(image_features, dim=0)
        text_matrix = torch.cat(text_features, dim=0)
        similarity = image_matrix @ text_matrix.T
        return RetrievalEvalOutput(
            image_to_text_recall_at_1=_paired_recall(similarity, 1, dim=1),
            image_to_text_recall_at_5=_paired_recall(similarity, 5, dim=1),
            image_to_text_recall_at_10=_paired_recall(similarity, 10, dim=1),
            text_to_image_recall_at_1=_paired_recall(similarity, 1, dim=0),
            text_to_image_recall_at_5=_paired_recall(similarity, 5, dim=0),
            text_to_image_recall_at_10=_paired_recall(similarity, 10, dim=0),
        )
