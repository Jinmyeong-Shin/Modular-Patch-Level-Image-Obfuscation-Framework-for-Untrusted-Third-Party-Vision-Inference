import os

from dataclasses import dataclass, KW_ONLY

from typing import Union, Optional

from transformers import AutoConfig

from .config import Config

@dataclass
class ModelConfig(Config):
    hf_model_name_or_path: Union[str, os.PathLike]
    hf_model_config: Optional[AutoConfig] = None

    def __post_init__(self):
        if self.hf_model_config is None:
            self.hf_model_config = AutoConfig.from_pretrained(self.hf_model_name_or_path)

@dataclass
class ImageModelConfig(ModelConfig):
    ...

@dataclass
class ImageModelConfigWithObfuscation(ImageModelConfig):
    obfuscation_patch_size: int = 14
    obfuscation_group_size: int = 100

@dataclass
class ImageClassificationModelConfig(ImageModelConfig):
    _: KW_ONLY
    num_classes: int

@dataclass
class ImageClassificationModelConfigWithObfuscation(ImageModelConfigWithObfuscation, ImageClassificationModelConfig):
    ...
    
    