import os
import tempfile

from vit_obfuscation.config.experiment import (
    ExperimentConfig,
    ModelConfig,
    ObfuscationConfig,
    TaskTrainingConfig,
)


def test_config_defaults():
    """ExperimentConfig should have sensible defaults."""
    config = ExperimentConfig()
    assert config.obfuscation.patch_size == 14
    assert config.obfuscation.group_size == 100
    assert config.obfuscation.apply_patch_permutation is False
    assert config.embedding_training.iterations == 1000
    assert TaskTrainingConfig().iterations == 300
    assert config.seed == 42


def test_config_from_yaml():
    """Should load config from YAML file."""
    yaml_content = """
name: test-experiment
model:
  hf_model_name_or_path: google/vit-base-patch16-224
  task: classification
  num_classes: 10
obfuscation:
  patch_size: 14
  group_size: 50
  apply_patch_permutation: true
embedding_training:
  iterations: 100
  learning_rate: 0.01
  batch_size: 16
dataset:
  hf_dataset_name_or_path: cifar10
  input_column: img
  label_column: label
seed: 123
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = ExperimentConfig.from_yaml(f.name)

    os.unlink(f.name)

    assert config.name == "test-experiment"
    assert config.model.hf_model_name_or_path == "google/vit-base-patch16-224"
    assert config.model.num_classes == 10
    assert config.obfuscation.group_size == 50
    assert config.obfuscation.apply_patch_permutation is True
    assert config.embedding_training.iterations == 100
    assert config.seed == 123


def test_config_task_training_optional():
    """task_training should be None when not specified."""
    yaml_content = """
name: zero-shot
model:
  hf_model_name_or_path: openai/clip-vit-base-patch32
  task: zero_shot_classification
dataset:
  hf_dataset_name_or_path: cifar10
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = ExperimentConfig.from_yaml(f.name)

    os.unlink(f.name)
    assert config.task_training is None


def test_revision_task_config_fields():
    """Revision task configs should support text/mask columns and eval limits."""
    yaml_content = """
name: revision-task
model:
  hf_model_name_or_path: openai/clip-vit-base-patch32
  task: image_text_retrieval
dataset:
  hf_dataset_name_or_path: AnyModal/flickr30k
  input_column: image
  text_column: caption
  label_column: caption
  mask_column: binary_mask
evaluation:
  batch_size: 8
  max_samples: 32
  max_new_tokens: 12
  num_beams: 2
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = ExperimentConfig.from_yaml(f.name)

    os.unlink(f.name)
    assert config.dataset.text_column == "caption"
    assert config.dataset.mask_column == "binary_mask"
    assert config.evaluation.max_samples == 32
    assert config.evaluation.max_new_tokens == 12


if __name__ == "__main__":
    test_config_defaults()
    test_config_from_yaml()
    test_config_task_training_optional()
    print("All config tests passed!")
