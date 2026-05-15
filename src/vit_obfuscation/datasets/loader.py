from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import datasets
from PIL import Image


DATASET_REPO_MAPPINGS = {
    "scene_parse_150": "zhoubolei/scene_parse_150",
}


def _resolve_hf_dataset_name(hf_dataset_name_or_path: str) -> str:
    return DATASET_REPO_MAPPINGS.get(hf_dataset_name_or_path, hf_dataset_name_or_path)


def _load_dataset_with_local_script(
    repo_id: str, split: str, subset: str | None = None
) -> datasets.Dataset:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "huggingface_hub is required to load dataset scripts for fallback datasets"
        ) from error

    script_filename = f"{Path(repo_id).name}.py"
    script_path = hf_hub_download(
        repo_id=repo_id, filename=script_filename, repo_type="dataset"
    )
    spec = importlib.util.spec_from_file_location(
        f"hf_dataset_script_{Path(repo_id).name}", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    builder_cls = getattr(module, "SceneParse150", None)
    if builder_cls is None:
        raise RuntimeError(f"Unable to import dataset builder from {script_path}")

    builder = builder_cls(name=subset) if subset is not None else builder_cls()
    builder.download_and_prepare()
    return builder.as_dataset(split=split)


def _load_dataset(
    hf_dataset_name_or_path: str,
    subset: str | None,
    split: str,
) -> datasets.Dataset:
    resolved_name = _resolve_hf_dataset_name(hf_dataset_name_or_path)
    try:
        if subset is not None:
            return datasets.load_dataset(resolved_name, name=subset, split=split)
        return datasets.load_dataset(resolved_name, split=split)
    except RuntimeError as exc:
        if (
            "Dataset scripts are no longer supported" in str(exc)
            and resolved_name == DATASET_REPO_MAPPINGS.get("scene_parse_150")
        ):
            return _load_dataset_with_local_script(
                resolved_name, split=split, subset=subset
            )
        raise


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


def _load_medmnist_as_hf(
    subset: str, split: str, input_column: str, label_column: str
) -> datasets.Dataset:
    """Load a MedMNIST dataset via the medmnist pip package and wrap as HF Dataset."""
    import medmnist

    cls_name = (
        subset.replace("mnist", "MNIST")
        .replace("path", "Path")
        .replace("derm", "Derm")
        .replace("blood", "Blood")
    )
    DatasetClass = getattr(medmnist, cls_name)
    ds = DatasetClass(split=split, download=True, size=28)

    images = []
    labels = []
    for i in range(len(ds)):
        img, label = ds[i]
        images.append(img.convert("RGB"))
        labels.append(int(label[0]))

    return datasets.Dataset.from_dict(
        {input_column: images, label_column: labels}
    ).cast_column(input_column, datasets.Image())


def load_classification_dataset(
    hf_dataset_name_or_path: str,
    input_column: str,
    train_split: str = "train",
    eval_split: str = "test",
    subset: str | None = None,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load a classification dataset with RGB conversion."""
    is_medmnist = "medmnist" in hf_dataset_name_or_path.lower() and subset is not None

    if is_medmnist:
        train_dataset = _load_medmnist_as_hf(subset, train_split, input_column, "label")
        eval_dataset = _load_medmnist_as_hf(subset, eval_split, input_column, "label")
    else:
        train_dataset = _load_dataset(
            hf_dataset_name_or_path, subset=subset, split=train_split
        )
        eval_dataset = _load_dataset(
            hf_dataset_name_or_path, subset=subset, split=eval_split
        )

    train_dataset.set_transform(lambda ex: _ensure_rgb(ex, input_column))
    eval_dataset.set_transform(lambda ex: _ensure_rgb(ex, input_column))

    return train_dataset, eval_dataset


def load_detection_dataset(
    hf_dataset_name_or_path: str,
    input_column: str,
    label_column: str,
    train_split: str = "train",
    eval_split: str = "validation",
    subset: str | None = None,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load an object detection dataset with RGB conversion and annotation reformatting."""

    def transform(ex):
        ex = _ensure_rgb(ex, input_column)
        ex = _reformat_detection_labels(ex, label_column)
        return ex

    train_dataset = _load_dataset(
        hf_dataset_name_or_path, subset=subset, split=train_split
    )
    train_dataset.set_transform(transform)

    eval_dataset = _load_dataset(
        hf_dataset_name_or_path, subset=subset, split=eval_split
    )
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
