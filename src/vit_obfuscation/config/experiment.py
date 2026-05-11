from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ObfuscationConfig:
    patch_size: int = 14
    group_size: int = 100


@dataclass
class ModelConfig:
    hf_model_name_or_path: str = ""
    task: str = "classification"
    num_classes: int | None = None
    id2label: dict[int, str] | None = None


@dataclass
class EmbeddingTrainingConfig:
    iterations: int = 5000
    learning_rate: float = 1e-2
    batch_size: int = 32
    training_dataset: str = "benjamin-paine/imagenet-1k-256x256"


@dataclass
class TaskTrainingConfig:
    iterations: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 256


@dataclass
class DatasetConfig:
    hf_dataset_name_or_path: str = ""
    subset: str | None = None
    input_column: str = "image"
    label_column: str = "label"
    train_split: str = "train"
    eval_split: str = "test"
    num_classes: int | None = None
    id2label: dict[int, str] | None = None


@dataclass
class EvaluationConfig:
    batch_size: int = 32
    with_obfuscation: bool = True


@dataclass
class ExperimentConfig:
    name: str = ""
    description: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    obfuscation: ObfuscationConfig = field(default_factory=ObfuscationConfig)
    embedding_training: EmbeddingTrainingConfig = field(
        default_factory=EmbeddingTrainingConfig
    )
    task_training: TaskTrainingConfig | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "./outputs"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> ExperimentConfig:
        config = cls()
        for key, value in data.items():
            if key == "model" and isinstance(value, dict):
                # Convert id2label keys to int
                if "id2label" in value and value["id2label"] is not None:
                    value["id2label"] = {
                        int(k): v for k, v in value["id2label"].items()
                    }
                config.model = ModelConfig(**value)
            elif key == "obfuscation" and isinstance(value, dict):
                config.obfuscation = ObfuscationConfig(**value)
            elif key == "embedding_training" and isinstance(value, dict):
                config.embedding_training = EmbeddingTrainingConfig(**value)
            elif key == "task_training" and isinstance(value, dict):
                if value is not None:
                    config.task_training = TaskTrainingConfig(**value)
            elif key == "dataset" and isinstance(value, dict):
                if "id2label" in value and value["id2label"] is not None:
                    value["id2label"] = {
                        int(k): v for k, v in value["id2label"].items()
                    }
                config.dataset = DatasetConfig(**value)
            elif key == "evaluation" and isinstance(value, dict):
                config.evaluation = EvaluationConfig(**value)
            else:
                setattr(config, key, value)
        return config
