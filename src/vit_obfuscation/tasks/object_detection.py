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
from .feature_utils import maybe_limit_dataset

_COCO_LABEL_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def _default_detection_id2label(dataset_name: str) -> dict[int, str] | None:
    if "coco" in dataset_name.lower():
        return {idx: name for idx, name in enumerate(_COCO_LABEL_NAMES)}
    return None


def _normalize_label_name(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


def _class_label_names(feature: Any) -> list[str] | None:
    if hasattr(feature, "names"):
        return list(feature.names)
    inner = getattr(feature, "feature", None)
    if inner is not None and hasattr(inner, "names"):
        return list(inner.names)
    return None


def _dataset_detection_id2label(dataset, label_column: str) -> dict[int, str] | None:
    features = getattr(dataset, "features", {})
    if not isinstance(features, dict) or label_column not in features:
        return None

    objects_feature = features[label_column]
    category_feature = None
    if isinstance(objects_feature, dict):
        category_feature = objects_feature.get("category") or objects_feature.get(
            "category_id"
        )
    elif hasattr(objects_feature, "__getitem__"):
        try:
            category_feature = objects_feature["category"]
        except (KeyError, TypeError):
            try:
                category_feature = objects_feature["category_id"]
            except (KeyError, TypeError):
                category_feature = None

    names = _class_label_names(category_feature)
    if names is None:
        return None
    return {idx: name for idx, name in enumerate(names)}


def _label_id_map(
    source_id2label: dict[int, str] | None,
    target_id2label: dict[int, str] | None,
) -> dict[int, int] | None:
    if not source_id2label or not target_id2label:
        return None

    target_by_name = {
        _normalize_label_name(name): int(idx)
        for idx, name in target_id2label.items()
        if not _normalize_label_name(name).startswith("n/a")
        and not _normalize_label_name(name).startswith("label ")
    }
    mapping = {
        int(source_idx): target_by_name[_normalize_label_name(source_name)]
        for source_idx, source_name in source_id2label.items()
        if _normalize_label_name(source_name) in target_by_name
    }
    return mapping or None


def _source_box_format(dataset_name: str) -> str:
    # detection-datasets/coco stores boxes as absolute [x0, y0, x1, y1].
    if dataset_name == "detection-datasets/coco":
        return "xyxy"
    return "auto"


def _infer_box_format(
    bbox: list[float],
    image_width: int | None = None,
    image_height: int | None = None,
    area: float | None = None,
) -> str:
    x0, y0, x2_or_w, y2_or_h = [float(v) for v in bbox]
    could_be_xyxy = x2_or_w > x0 and y2_or_h > y0
    if not could_be_xyxy:
        return "xywh"

    if area is not None:
        xyxy_area = max(0.0, x2_or_w - x0) * max(0.0, y2_or_h - y0)
        xywh_area = max(0.0, x2_or_w) * max(0.0, y2_or_h)
        if abs(float(area) - xyxy_area) < abs(float(area) - xywh_area):
            return "xyxy"
        if abs(float(area) - xywh_area) < abs(float(area) - xyxy_area):
            return "xywh"

    if image_width is not None and image_height is not None:
        xyxy_in_bounds = x2_or_w <= image_width and y2_or_h <= image_height
        xywh_in_bounds = x0 + x2_or_w <= image_width and y0 + y2_or_h <= image_height
        if xyxy_in_bounds and not xywh_in_bounds:
            return "xyxy"

    return "xywh"


def _box_to_xywh(
    bbox: list[float],
    image_width: int | None = None,
    image_height: int | None = None,
    area: float | None = None,
    source_format: str = "auto",
) -> list[float]:
    if source_format == "auto":
        source_format = _infer_box_format(bbox, image_width, image_height, area)

    x0, y0, x2_or_w, y2_or_h = [float(v) for v in bbox]
    if source_format == "xyxy":
        return [x0, y0, max(0.0, x2_or_w - x0), max(0.0, y2_or_h - y0)]
    if source_format == "xywh":
        return [x0, y0, max(0.0, x2_or_w), max(0.0, y2_or_h)]
    raise ValueError(f"Unsupported box format: {source_format}")


def _xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def _normalize_coco_annotation(
    annotation: dict,
    image_width: int | None = None,
    image_height: int | None = None,
    *,
    category_id_map: dict[int, int] | None = None,
    box_format: str = "auto",
) -> dict:
    normalized = {}

    bbox = annotation.get("bbox")
    if bbox is not None:
        bbox = list(bbox)
        if len(bbox) == 4:
            bbox = _box_to_xywh(
                bbox,
                image_width=image_width,
                image_height=image_height,
                area=annotation.get("area"),
                source_format=box_format,
            )
        normalized["bbox"] = bbox
    if "category_id" in annotation:
        category_id = annotation["category_id"]
    elif "category" in annotation:
        category_id = annotation["category"]
    else:
        category_id = None
    if category_id is not None:
        category_id = int(category_id)
        if category_id_map is not None:
            category_id = category_id_map.get(category_id, category_id)
        normalized["category_id"] = category_id
    if "area" in annotation:
        normalized["area"] = annotation["area"]
    elif "bbox" in normalized:
        _, _, w, h = normalized["bbox"]
        normalized["area"] = float(w) * float(h)
    if "iscrowd" in annotation:
        normalized["iscrowd"] = annotation["iscrowd"]
    if "keypoints" in annotation:
        normalized["keypoints"] = annotation["keypoints"]
    return normalized


def _normalize_detection_labels(
    labels,
    image_ids=None,
    image_sizes=None,
    *,
    category_id_map: dict[int, int] | None = None,
    box_format: str = "auto",
):
    if labels is None:
        return None

    if isinstance(labels, str):
        labels = json.loads(labels)

    if isinstance(labels, dict):
        labels = [labels]

    normalized_batch = []
    for idx, label_data in enumerate(labels):
        image_id = None
        if image_ids is not None and idx < len(image_ids):
            image_id = image_ids[idx]

        if isinstance(label_data, str):
            label_data = json.loads(label_data)

        if isinstance(label_data, dict) and "bbox" in label_data and not isinstance(
            label_data["bbox"][0], list
        ):
            annos = [
                _normalize_coco_annotation(
                    label_data,
                    *(image_sizes[idx] if image_sizes else (None, None)),
                    category_id_map=category_id_map,
                    box_format=box_format,
                )
            ]
        elif isinstance(label_data, dict):
            list_of_annos = _reformat_single_detection(label_data)
            annos = [
                _normalize_coco_annotation(
                    anno,
                    *(image_sizes[idx] if image_sizes else (None, None)),
                    category_id_map=category_id_map,
                    box_format=box_format,
                )
                for anno in list_of_annos
            ]
        elif isinstance(label_data, list):
            annos = [
                _normalize_coco_annotation(
                    anno,
                    *(image_sizes[idx] if image_sizes else (None, None)),
                    category_id_map=category_id_map,
                    box_format=box_format,
                )
                for anno in label_data
            ]
        else:
            annos = label_data

        normalized_batch.append({"image_id": image_id, "annotations": annos})

    return normalized_batch


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
        image_size = getattr(self.obfuscator, "image_size", None)
        if isinstance(image_size, tuple):
            return {"size": {"height": image_size[0], "width": image_size[1]}}
        if image_size is not None:
            return {"size": image_size}
        return {}

    def _category_id_map(self, dataset) -> dict[int, int] | None:
        source_id2label = (
            self.config.dataset.id2label
            or _dataset_detection_id2label(dataset, self.config.dataset.label_column)
            or _default_detection_id2label(self.config.dataset.hf_dataset_name_or_path)
        )
        target_id2label = getattr(self.adapter.model.config, "id2label", None)
        return _label_id_map(source_id2label, target_id2label)

    def forward(
        self,
        images,
        labels=None,
        image_ids=None,
        image_sizes=None,
        with_obfuscation: bool = False,
        category_id_map: dict[int, int] | None = None,
        box_format: str = "auto",
        **kwargs,
    ):
        model = self.adapter.model
        proc_kwargs = self._get_processor_kwargs()

        if labels is not None:
            annotations = _normalize_detection_labels(
                labels,
                image_ids=image_ids,
                image_sizes=image_sizes,
                category_id_map=category_id_map,
                box_format=box_format,
            )
            inputs = self.processor(
                images=images,
                annotations=annotations,
                return_tensors="pt",
                **proc_kwargs,
            )
        else:
            inputs = self.processor(images=images, return_tensors="pt", **proc_kwargs)

        device = next(model.parameters()).device
        inputs = self._move_to_device(inputs, device)

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
        category_id_map = self._category_id_map(train_dataset)
        box_format = _source_box_format(cfg.dataset.hf_dataset_name_or_path)

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
            image_ids = []
            image_sizes = []
            for item in batch:
                label_data = item[cfg.dataset.label_column]
                if isinstance(label_data, str):
                    label_data = json.loads(label_data)
                processed_labels.append(label_data)
                image_ids.append(item.get("image_id"))
                image_sizes.append((item.get("width"), item.get("height")))
            return {
                cfg.dataset.input_column: [
                    item[cfg.dataset.input_column] for item in batch
                ],
                cfg.dataset.label_column: processed_labels,
                "image_ids": image_ids,
                "image_sizes": image_sizes,
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
                image_ids=batch.get("image_ids"),
                image_sizes=batch.get("image_sizes"),
                category_id_map=category_id_map,
                box_format=box_format,
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
        eval_dataset = maybe_limit_dataset(eval_dataset, cfg.evaluation.max_samples)
        category_id_map = self._category_id_map(eval_dataset)
        box_format = _source_box_format(cfg.dataset.hf_dataset_name_or_path)

        def collate_fn(batch):
            processed_labels = []
            for item in batch:
                label_data = item[cfg.dataset.label_column]
                if isinstance(label_data, str):
                    label_data = json.loads(label_data)
                normalized = _normalize_detection_labels(
                    [label_data],
                    image_sizes=[(item.get("width"), item.get("height"))],
                    category_id_map=category_id_map,
                    box_format=box_format,
                )
                processed_labels.append(normalized[0]["annotations"])
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
                        boxes.append(_xywh_to_xyxy(box))
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

    def _get_processor_kwargs(self) -> dict:
        image_size = getattr(self.obfuscator, "image_size", None)
        if isinstance(image_size, tuple):
            return {"size": {"height": image_size[0], "width": image_size[1]}}
        if image_size is not None:
            return {"size": image_size}
        return {}

    def forward(self, images, text=None, with_obfuscation: bool = False, **kwargs):
        model = self.adapter.model
        proc_kwargs = self._get_processor_kwargs()
        inputs = self.processor(
            images=images,
            text=text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **proc_kwargs,
        )
        device = next(model.parameters()).device
        inputs = self._move_to_device(inputs, device)

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
        eval_dataset = maybe_limit_dataset(eval_dataset, cfg.evaluation.max_samples)

        id2label = cfg.dataset.id2label or cfg.model.id2label
        if id2label is None:
            id2label = _dataset_detection_id2label(
                eval_dataset, cfg.dataset.label_column
            )
        if id2label is None:
            id2label = _default_detection_id2label(cfg.dataset.hf_dataset_name_or_path)

        if id2label is None:
            raise ValueError("id2label mapping required for zero-shot detection")
        label_names = [name for _, name in sorted(id2label.items())]
        box_format = _source_box_format(cfg.dataset.hf_dataset_name_or_path)

        def collate_fn(batch):
            processed_labels = []
            for item in batch:
                label_data = item[cfg.dataset.label_column]
                if isinstance(label_data, str):
                    label_data = json.loads(label_data)
                normalized = _normalize_detection_labels(
                    [label_data],
                    image_sizes=[(item.get("width"), item.get("height"))],
                    box_format=box_format,
                )
                processed_labels.append(normalized[0]["annotations"])
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
        proc_kwargs = self._get_processor_kwargs()
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
                    **proc_kwargs,
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
                        boxes.append(_xywh_to_xyxy(box))
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
