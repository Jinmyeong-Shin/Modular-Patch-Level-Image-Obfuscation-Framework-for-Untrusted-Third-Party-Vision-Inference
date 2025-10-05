from dataclasses import dataclass
from typing import TypeVar, Generic, Optional

import numpy as np

import torch

from transformers.modeling_outputs import BaseModelOutput

from .output import Output

TransformersModelOutput_T = TypeVar('TransformersModelOutput_T', bound=BaseModelOutput)

@dataclass
class ModelOutput(Generic[TransformersModelOutput_T], Output):
    model_outputs: Optional[TransformersModelOutput_T] = None

@dataclass
class ModelOutputWithObfuscation(Output, Generic[TransformersModelOutput_T]):
    obfuscated_images: Optional[torch.FloatTensor] = None
    model_outputs: Optional[TransformersModelOutput_T] = None

@dataclass
class ImageClassificationModelOutputWithObfuscation(ModelOutputWithObfuscation[TransformersModelOutput_T]):
    logits: Optional[torch.FloatTensor] = None

@dataclass
class ObjectDetectionModelOutputWithObfuscation(ModelOutputWithObfuscation):
    """
    Represents the output of an object detection model, including obfuscated images if any.
    This should be defined in `data/model_output.py`.
    """
    pass