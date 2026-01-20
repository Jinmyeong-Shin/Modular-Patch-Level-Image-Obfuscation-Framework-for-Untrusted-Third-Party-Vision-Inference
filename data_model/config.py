from typing import Optional, Tuple, Type, TypeVar, Union

from dataclasses import dataclass

import yaml

from pathlib import Path

from .data import BaseModel

_T = TypeVar('_T', bound='BaseConfig')

class BaseConfig(BaseModel):
    """
    Base configuration class that integrates with BaseModel for serialization
    and provides a loader from YAML files.
    
    Subclasses are expected to be dataclasses that inherit from this class.
    This provides a structured, type-safe way to manage configuration.
    """

    @classmethod
    def from_yaml(cls: Type[_T], config_path: str) -> _T:
        """
        Creates a config instance by loading parameters from the specified YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f'Configuration file not found: {path}')
        if path.suffix not in ['.yaml', '.yml']:
            raise ValueError(f'Configuration file must be a YAML file (.yaml or .yml): {path}')
        
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            raise TypeError(f'YAML file {path} did not load as a dictionary.')

        # Use the from_dict method inherited from BaseModel
        return cls.from_dict(config_data)
    
@dataclass
class ObfuscatorConfig(BaseConfig):
    """
    Configuration for the Obfuscator layer.

    Args:
        image_size (Union[int, Tuple[int, int]]): The size of the input image.
        num_channels (int): The number of channels in the input image.
        patch_size (Union[int, Tuple[int, int]]): The size of the patches to be obfuscated.
        num_kernels (int): The number of distinct kernels (permutation matrices) per channel.
        kernels_path (Optional[str]): Path to a .safetensors file to load pre-existing
            kernels. If None, kernels will be initialized randomly.
    """
    image_size: Union[int, Tuple[int, int]]
    num_channels: int
    patch_size: Union[int, Tuple[int, int]]
    num_kernels: int
    kernels_path: Optional[str] = None

@dataclass
class DeobfuscatorConfig(BaseConfig):
    """
    Configuration for the Deobfuscator layer.

    Args:
        image_size (Union[int, Tuple[int, int]]): The size of the input image.
        num_channels (int): The number of channels in the input image from the obfuscator.
        patch_size (Union[int, Tuple[int, int]]): The size of the patches.
        embed_dim (int): The dimension of the transformer's embedding.
        add_cls_token (bool): Whether to add a learnable CLS token. Defaults to True.
        add_position_embeddings (bool): Whether to add learnable position embeddings. Defaults to True.
        num_extra_tokens (int): Number of additional learnable tokens (e.g., detection tokens for YOLOS). Defaults to 0.
        weights_path (Optional[str]): Path to a .safetensors file to load pre-existing
            weights for the patch embedding. If None, weights will be initialized randomly.
    """
    image_size: Union[int, Tuple[int, int]]
    num_channels: int
    patch_size: Union[int, Tuple[int, int]]
    embed_dim: int
    add_cls_token: bool = True
    add_position_embeddings: bool = True
    num_extra_tokens: int = 0
    weights_path: Optional[str] = None