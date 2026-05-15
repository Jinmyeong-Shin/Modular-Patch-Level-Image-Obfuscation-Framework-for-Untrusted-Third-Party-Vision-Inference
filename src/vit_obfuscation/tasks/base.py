from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

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

    def _move_to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, Mapping):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v, device) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v, device) for v in obj)
        return obj

    def process_images(self, images, **processor_kwargs) -> dict:
        """Process raw images through the model's processor."""
        return self.processor(images=images, return_tensors="pt", **processor_kwargs)

    def task_state_dict(self) -> dict[str, Any]:
        """Return trainable task state for checkpointing."""
        state: dict[str, Any] = {}
        model = getattr(self.adapter, "model", None)
        if isinstance(model, nn.Module):
            state["model"] = model.state_dict()

        for name, value in self.__dict__.items():
            if name in {"adapter", "obfuscator", "obf_embedding"}:
                continue
            if isinstance(value, nn.Module):
                state[name] = value.state_dict()
        return state

    def load_task_state_dict(self, state: dict[str, Any]) -> None:
        """Load trainable task state saved by task_state_dict."""
        model_state = state.get("model")
        model = getattr(self.adapter, "model", None)
        if model_state is not None and isinstance(model, nn.Module):
            model.load_state_dict(model_state)

        for name, value in self.__dict__.items():
            if name in {"adapter", "obfuscator", "obf_embedding"}:
                continue
            if isinstance(value, nn.Module) and name in state:
                value.load_state_dict(state[name])

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
