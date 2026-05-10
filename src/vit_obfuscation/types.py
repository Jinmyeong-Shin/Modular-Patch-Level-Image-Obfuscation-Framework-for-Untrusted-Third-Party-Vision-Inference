from typing import Union

import numpy as np
import PIL
import torch

ImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    list[PIL.Image.Image],
    list[np.ndarray],
    list[torch.Tensor],
]
