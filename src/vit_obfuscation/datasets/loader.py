from __future__ import annotations

import datasets
from PIL import Image


def _ensure_rgb(example: dict, input_column: str) -> dict:
    """Convert images to RGB format."""
    img = example[input_column]
    if isinstance(img, list):
        example[input_column] = [
            im.convert("RGB") if isinstance(im, Image.Image) else im for im in img
        ]
    elif isinstance(img, Image.Image):
        example[input_column] = img.convert("RGB")
    return example


def _reformat_detection_labels(example: dict, label_column: str) -> dict:
    """
    Reformat object detection annotations from HF dataset format
    (dict of lists) to list of dicts (expected by HF processors).
    """
    objects = example.get(label_column)
    if objects is None:
        return example

    if isinstance(objects, list):
        # Handle batched case
        processed = []
        for objs in objects:
            processed.append(_reformat_single_detection(objs))
        example[label_column] = processed
    else:
        example[label_column] = _reformat_single_detection(objects)
    return example


def _reformat_single_detection(objects) -> list[dict]:
    """Convert a single detection annotation from dict-of-lists to list-of-dicts."""
    if isinstance(objects, list):
        return objects
    if not isinstance(objects, dict) or "bbox" not in objects:
        return objects

    annotations = []
    num_objects = len(objects["bbox"])
    for i in range(num_objects):
        annotation = {}
        for key, value_list in objects.items():
            if isinstance(value_list, list) and i < len(value_list):
                annotation[key] = value_list[i]
        annotations.append(annotation)
    return annotations


def load_classification_dataset(
    hf_dataset_name_or_path: str,
    input_column: str,
    train_split: str = "train",
    eval_split: str = "test",
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load a classification dataset with RGB conversion."""
    train_dataset = datasets.load_dataset(hf_dataset_name_or_path, split=train_split)
    train_dataset.set_transform(lambda ex: _ensure_rgb(ex, input_column))

    eval_dataset = datasets.load_dataset(hf_dataset_name_or_path, split=eval_split)
    eval_dataset.set_transform(lambda ex: _ensure_rgb(ex, input_column))

    return train_dataset, eval_dataset


def load_detection_dataset(
    hf_dataset_name_or_path: str,
    input_column: str,
    label_column: str,
    train_split: str = "train",
    eval_split: str = "validation",
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load an object detection dataset with RGB conversion and annotation reformatting."""

    def transform(ex):
        ex = _ensure_rgb(ex, input_column)
        ex = _reformat_detection_labels(ex, label_column)
        return ex

    train_dataset = datasets.load_dataset(hf_dataset_name_or_path, split=train_split)
    train_dataset.set_transform(transform)

    eval_dataset = datasets.load_dataset(hf_dataset_name_or_path, split=eval_split)
    eval_dataset.set_transform(transform)

    return train_dataset, eval_dataset


def load_embedding_training_dataset(
    hf_dataset_name_or_path: str = "benjamin-paine/imagenet-1k-256x256",
    split: str = "train",
) -> datasets.Dataset:
    """Load dataset for embedding training (default: ImageNet-1k-256)."""
    dataset = datasets.load_dataset(hf_dataset_name_or_path, split=split)
    dataset.set_transform(lambda ex: _ensure_rgb(ex, "image"))
    return dataset
