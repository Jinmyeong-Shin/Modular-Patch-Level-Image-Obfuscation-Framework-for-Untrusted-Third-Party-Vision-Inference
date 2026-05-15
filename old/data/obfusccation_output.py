from dataclasses import dataclass
from typing import Optional

import PIL

import numpy as np

import torch

from transformers.image_processing_base import BatchFeature

from .output import Output

@dataclass
class ObfuscationOutput(Output):
    obfuscated_images: Optional[torch.FloatTensor] = None,
    processed_inputs: Optional[BatchFeature] = None