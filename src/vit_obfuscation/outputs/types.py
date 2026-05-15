from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from typing import Any, Iterable, Optional

import torch

try:
    import torch.utils._pytree as _torch_pytree
except ImportError:
    _torch_pytree = None


def _output_flatten(output):
    return list(output.values()), list(output.keys())


def _output_unflatten(values, context, output_type=None):
    return output_type(**dict(zip(context, values)))


class Output(OrderedDict):
    """
    Base class for all outputs as dataclass. Supports both dict-style
    and tuple-style access. Compatible with PyTorch pytree utilities.
    """

    def __init_subclass__(cls) -> None:
        if _torch_pytree is not None:
            _torch_pytree.register_pytree_node(
                cls,
                _output_flatten,
                partial(_output_unflatten, output_type=cls),
                serialized_type_name=f"{cls.__module__}.{cls.__name__}",
            )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.__class__ != Output and not is_dataclass(self):
            raise TypeError(
                f"{self.__class__.__name__} must use the @dataclass decorator."
            )

    def __post_init__(self) -> None:
        # Populate OrderedDict from dataclass fields after they are set
        for field in fields(self):
            v = getattr(self, field.name)
            if v is not None:
                OrderedDict.__setitem__(self, field.name, v)

    def __getitem__(self, k):
        if isinstance(k, str):
            return dict(self.items())[k]
        return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] for k in self.keys())


@dataclass
class ObfuscationOutput(Output):
    obfuscated_images: Optional[torch.FloatTensor] = None
    processed_inputs: Optional[Any] = None


@dataclass
class ModelOutputWithObfuscation(Output):
    obfuscated_images: Optional[torch.FloatTensor] = None
    model_outputs: Optional[Any] = None


@dataclass
class ClassificationModelOutput(ModelOutputWithObfuscation):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class ClassificationEvalOutput(Output):
    accuracy: Optional[float] = None
    recall: Optional[float] = None
    precision: Optional[float] = None
    f1: Optional[float] = None


@dataclass
class DetectionEvalOutput(Output):
    map: Optional[float] = None
    map_50: Optional[float] = None
    map_75: Optional[float] = None
    map_small: Optional[float] = None
    map_medium: Optional[float] = None
    map_large: Optional[float] = None
    mar_1: Optional[float] = None
    mar_10: Optional[float] = None
    mar_100: Optional[float] = None
    mar_small: Optional[float] = None
    mar_medium: Optional[float] = None
    mar_large: Optional[float] = None


@dataclass
class SegmentationEvalOutput(Output):
    mean_iou: Optional[float] = None
    mean_accuracy: Optional[float] = None
    overall_accuracy: Optional[float] = None
    per_category_iou: Optional[dict[str, float]] = None
    per_category_accuracy: Optional[dict[str, float]] = None


@dataclass
class RetrievalEvalOutput(Output):
    recall_at_1: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    map: Optional[float] = None
    image_to_text_recall_at_1: Optional[float] = None
    image_to_text_recall_at_5: Optional[float] = None
    image_to_text_recall_at_10: Optional[float] = None
    text_to_image_recall_at_1: Optional[float] = None
    text_to_image_recall_at_5: Optional[float] = None
    text_to_image_recall_at_10: Optional[float] = None


@dataclass
class AnomalyEvalOutput(Output):
    image_auroc: Optional[float] = None
    image_average_precision: Optional[float] = None
    pixel_auroc: Optional[float] = None
    pro_score: Optional[float] = None


@dataclass
class BinarySegmentationEvalOutput(Output):
    dice: Optional[float] = None
    iou: Optional[float] = None
    pixel_accuracy: Optional[float] = None


@dataclass
class CaptioningEvalOutput(Output):
    bleu1: Optional[float] = None
    bleu4: Optional[float] = None
    exact_match: Optional[float] = None
