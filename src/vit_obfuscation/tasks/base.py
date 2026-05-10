from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..adapter.model_adapter import ModelAdapter
from ..config.experiment import ExperimentConfig
from ..embedding.embedding import ObfuscationEmbedding
from ..obfuscation.obfuscator import ChannelWisePatchLevelObfuscator
from ..outputs.types import ObfuscationOutput


class BaseTask(ABC):
    """
    Abstract base class for all vision tasks.
    Tasks receive a ModelAdapter and obfuscation modules - they don't own the model.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        obfuscator: ChannelWisePatchLevelObfuscator,
        obf_embedding: ObfuscationEmbedding,
        processor,
        config: ExperimentConfig,
    ) -> None:
        self.adapter = adapter
        self.obfuscator = obfuscator
        self.obf_embedding = obf_embedding
        self.processor = processor
        self.config = config

    def obfuscate(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Apply obfuscation to preprocessed pixel values."""
        return self.obfuscator(pixel_values)

    def obfuscation_modules_to(self, device) -> None:
        """Move obfuscation modules to the specified device."""
        self.obfuscator = self.obfuscator.to(device)
        self.obf_embedding = self.obf_embedding.to(device)

    def process_images(self, images, **processor_kwargs) -> dict:
        """Process raw images through the model's processor."""
        return self.processor(images=images, return_tensors="pt", **processor_kwargs)

    @abstractmethod
    def forward(self, images, with_obfuscation: bool = False, **kwargs) -> Any:
        """Run forward pass, optionally with obfuscation."""
        ...

    @abstractmethod
    def train_task(self) -> None:
        """Train task-specific components (e.g., classifier head)."""
        ...

    @abstractmethod
    def evaluate(self, with_obfuscation: bool = False) -> Any:
        """Evaluate on the configured dataset."""
        ...
