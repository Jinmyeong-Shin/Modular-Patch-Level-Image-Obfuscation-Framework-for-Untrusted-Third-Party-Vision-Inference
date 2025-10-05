from typing import Union

import PIL

import numpy as np

import torch

ImageInput = Union[
    PIL.Image.Image, np.ndarray, torch.Tensor, list[PIL.Image.Image], list[np.ndarray], list[torch.Tensor]
]