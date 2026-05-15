from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def maybe_limit_dataset(dataset, max_samples: int | None):
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return dataset.select(range(max_samples))


def collate_images(batch: list[dict], input_column: str) -> list[Any]:
    return [item[input_column] for item in batch]


def first_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("caption", "text", "raw", "sentence"):
            if key in value:
                return first_text(value[key])
    if isinstance(value, Iterable):
        for item in value:
            text = first_text(item)
            if text:
                return text
    return "" if value is None else str(value)


def normalize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def labels_to_tensor(labels: list[Any]) -> torch.Tensor:
    encoded = []
    lookup: dict[str, int] = {}
    for label in labels:
        if isinstance(label, torch.Tensor):
            label = label.item()
        if isinstance(label, (int, np.integer)):
            encoded.append(int(label))
        else:
            key = str(label)
            if key not in lookup:
                lookup[key] = len(lookup)
            encoded.append(lookup[key])
    return torch.tensor(encoded, dtype=torch.long)


def image_to_binary_mask(mask: Any, size: tuple[int, int] | None = None) -> torch.Tensor:
    if mask is None:
        raise ValueError("Mask value is required for binary segmentation metrics")
    if isinstance(mask, Image.Image):
        arr = np.array(mask)
    elif isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.array(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    tensor = torch.from_numpy((arr > 0).astype("float32")).unsqueeze(0)
    if size is not None and tuple(tensor.shape[-2:]) != tuple(size):
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=size, mode="nearest"
        ).squeeze(0)
    return tensor


def output_patch_tokens(outputs) -> torch.Tensor:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise ValueError("Model output does not expose last_hidden_state")
    seq_len = hidden.shape[1]
    patch_count = seq_len - 1
    side = int(patch_count**0.5)
    if side * side == patch_count:
        return hidden[:, 1:, :]
    side = int(seq_len**0.5)
    if side * side == seq_len:
        return hidden
    return hidden[:, 1:, :] if seq_len > 1 else hidden


def pooled_features_from_outputs(outputs) -> torch.Tensor:
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0] if hidden.shape[1] > 1 else hidden.mean(dim=1)
    image_embeds = getattr(outputs, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds
    raise ValueError("Unable to derive pooled image features from model output")


def vision_forward(model, pixel_values: torch.Tensor):
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        return vision_model(pixel_values=pixel_values)
    clip = getattr(model, "clip", None)
    if clip is not None and getattr(clip, "vision_model", None) is not None:
        return clip.vision_model(pixel_values=pixel_values)
    return model(pixel_values=pixel_values)


def encode_image_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=pixel_values)
    return pooled_features_from_outputs(vision_forward(model, pixel_values))


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        features = pooled_features_from_outputs(features)
    return F.normalize(features.float(), dim=-1)
