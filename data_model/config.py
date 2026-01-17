import yaml

from pathlib import Path

class BaseConfig:
    """
    Base configuration class that reads parameters from a YAML file.
    This class is designed to be extended for specific configurations
    like DL model configurations (compatible with Hugging Face Transformers)
    and dataset configurations.
    """

    def __init__(self, config_path: str):
        """
        Initializes the BaseConfig by loading parameters from the specified YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        if not self.config_path.suffix == '.yaml':
            raise ValueError(f"Configuration file must be a YAML file (.yaml): {self.config_path}")

        self._load_config()

    def _load_config(self):
        """
        Loads the configuration parameters from the YAML file into the object's attributes.
        """
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        for key, value in config_data.items():
            setattr(self, key, value)