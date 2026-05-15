from dataclasses import dataclass
from typing import TypeVar, Generic, Optional

import numpy as np

import torch

from transformers.modeling_outputs import BaseModelOutput

from .output import Output

BaseModelOutput_T = TypeVar('BaseModelOutput_T', bound=BaseModelOutput)

@dataclass
class EvaluationOutput(Output, Generic[BaseModelOutput_T]):
    model_outputs: Optional[BaseModelOutput_T] = None


@dataclass
class ClassificationEvaluationOutput(EvaluationOutput):
    logits: Optional[torch.FloatTensor] = None
    accuracy: Optional[float] = None,
    recall: Optional[float] = None,
    precision: Optional[float] = None,
    f1: Optional[float] = None,
    
@dataclass
class ObjectDetectionEvaluationOutput(EvaluationOutput):
    """
    Represents the evaluation output for an object detection task.
    This should be defined in `data/evaluation_output.py`.
    NOTE: This implementation requires `torchmetrics` to be installed (`pip install torchmetrics`).
    """
    map: Optional[float] = None
    map_50: Optional[float] = None
    map_75: Optional[float] = None
    map_small: Optional[float] = None
    map_medium: Optional[float] = None
    map_large: Optional[float] = None
    # The 'mar' stands for mean average recall.
    mar_1: Optional[float] = None
    mar_10: Optional[float] = None
    mar_100: Optional[float] = None
    mar_small: Optional[float] = None
    mar_medium: Optional[float] = None
    mar_large: Optional[float] = None