import os
import inspect

import yaml

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Union, Optional

from .config import Config

import datasets

_DATASET_CONFIG_TYPE_REGISTRY: dict[str, type] = {
    'image_classification': 'ImageClassificationDatasetConfigWithTrainingAndEvaluationSplits',
    'object_detection': 'ObjectDetectionDatasetConfigWithTrainingAndEvaluationSplits',
}

@dataclass(kw_only=True)
class DatasetConfig(ABC, Config):
    name: str

    hf_dataset_name_or_path: Union[str, os.PathLike]

    @abstractmethod
    def build(self) -> tuple[datasets.Dataset]:
        ...

@dataclass(kw_only=True)
class DatasetConfigWithSplits(DatasetConfig):
    splits: list[str]


@dataclass(kw_only=True)
class DatasetConfigWithTrainingAndEvaluationSplits(DatasetConfigWithSplits):
    train_split: str
    evaluation_split: str

@dataclass(kw_only=True)
class ClassificationDatasetConfig(DatasetConfig):
    input_column: str
    label_column: str

    num_classes: int
    id2label: Optional[dict[int, str]] = None

    def build(self) -> Union[datasets.Dataset, datasets.DatasetDict, datasets.IterableDataset, datasets.IterableDatasetDict]:
        return datasets.load_dataset(self.hf_dataset_name_or_path)


@dataclass
class ImageClassificationDatasetConfig(ClassificationDatasetConfig):
    def build(self) -> tuple[datasets.Dataset, datasets.Dataset]:
        dataset = datasets.load_dataset(self.hf_dataset_name_or_path)
        dataset.set_transform(self.preprocess)

        return dataset

    def preprocess(self, batch):
        from PIL import Image
        if isinstance(batch[self.input_column], list):
            batch[self.input_column] = [img.convert('RGB') if isinstance(img, Image.Image) else img for img in batch[self.input_column]]
        else:
            batch[self.input_column] = batch[self.input_column].convert('RGB') if isinstance(batch[self.input_column], Image.Image) else batch[self.input_column]
        return batch
    
    
@dataclass
class ObjectDetectionDatasetConfig(ClassificationDatasetConfig):
    def build(self) -> tuple[datasets.Dataset, datasets.Dataset]:
        dataset = datasets.load_dataset(self.hf_dataset_name_or_path)
        dataset.set_transform(self.preprocess)

        return dataset

    def preprocess(self, batch):
        from PIL import Image
        if self.input_column in batch:
            if isinstance(batch[self.input_column], list):
                batch[self.input_column] = [img.convert('RGB') if isinstance(img, Image.Image) else img for img in batch[self.input_column]]
            else:
                batch[self.input_column] = batch[self.input_column].convert('RGB') if isinstance(batch[self.input_column], Image.Image) else batch[self.input_column]

        # Reformat annotations from dataset's format to a list of dicts,
        # which is expected by Hugging Face processors.
        if self.label_column in batch:
            processed_labels = []
            for objects in batch[self.label_column]:
                annotations = []
                if objects and 'bbox' in objects and isinstance(objects['bbox'], list):
                    num_objects = len(objects['bbox'])
                    for i in range(num_objects):
                        annotation = {}
                        for key, value_list in objects.items():
                            if isinstance(value_list, list) and i < len(value_list):
                                annotation[key] = value_list[i]
                        annotations.append(annotation)
                processed_labels.append(annotations)
            batch[self.label_column] = processed_labels
        return batch

@dataclass
class ObjectDetectionDatasetConfigWithTrainingAndEvaluationSplits(DatasetConfigWithTrainingAndEvaluationSplits, ObjectDetectionDatasetConfig):
    def build(self) -> Union[tuple[datasets.Dataset, datasets.Dataset], tuple[datasets.IterableDataset, datasets.IterableDataset]]:
        train_dataset = datasets.load_dataset(
            self.hf_dataset_name_or_path,
            split=self.train_split,
        )
        train_dataset.set_transform(self.preprocess, output_all_columns=True)

        eval_dataset = datasets.load_dataset(
            self.hf_dataset_name_or_path,
            split=self.evaluation_split,
        )
        eval_dataset.set_transform(self.preprocess, output_all_columns=True)

        return train_dataset, eval_dataset
    
@dataclass
class ImageClassificationDatasetConfigWithTrainingAndEvaluationSplits(DatasetConfigWithTrainingAndEvaluationSplits, ImageClassificationDatasetConfig):

    def build(self) -> Union[tuple[datasets.Dataset, datasets.Dataset], tuple[datasets.IterableDataset, datasets.IterableDataset]]:
        train_dataset = datasets.load_dataset(
            self.hf_dataset_name_or_path,
            split=self.train_split,
        )
        train_dataset.set_transform(self.preprocess)
        train_dataset.set_format('torch')

        eval_dataset = datasets.load_dataset(
            self.hf_dataset_name_or_path,
            split=self.evaluation_split,
        )
        eval_dataset.set_transform(self.preprocess)
        eval_dataset.set_format('torch')

        return train_dataset, eval_dataset


def load_dataset_config(name_or_path: Union[str, os.PathLike]) -> DatasetConfig:
    """
    Loads a dataset configuration from a YAML file.

    The YAML file must contain a 'dataset_config_class' key that specifies
    the name of the DatasetConfig subclass to use.

    Args:
        name_or_path (Union[str, os.PathLike]): The path to the YAML configuration file.

    Returns:
        DatasetConfig: An instance of a DatasetConfig subclass.
    """
    config_path = None
    if os.path.exists(name_or_path):
        config_path = name_or_path
    else:
        # If not, assume it's a name and construct the path relative to this script's location.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, "..", "configs", "dataset", f"{name_or_path}.yaml")
        if os.path.exists(potential_path):
            config_path = potential_path

    if config_path is None:
        raise FileNotFoundError(f"Dataconfig file not found for '{name_or_path}'")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    type_name = config_data.pop('type', None)
    if type_name is None:
        raise ValueError("Dataconfig file must contain a 'type' key.")

    class_name = _DATASET_CONFIG_TYPE_REGISTRY.get(type_name)
    if class_name is None:
        raise ValueError(
            f"Unknown dataset config type '{type_name}'. "
            f"Available types are: {list(_DATASET_CONFIG_TYPE_REGISTRY.keys())}"
        )

    config_class = globals().get(class_name)

    if config_class is None:
        raise TypeError(f"Dataset config class '{class_name}' not found for type '{type_name}'.")

    if not inspect.isclass(config_class) or not issubclass(config_class, DatasetConfig):
        raise TypeError(f"'{class_name}' is not a valid DatasetConfig subclass.")

    return config_class(**config_data)