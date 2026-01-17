from abc import ABC, abstractmethod

from typing import Any, Dict

class BaseModel(ABC):
    """
    Abstract Base Class for data models used as input/output interfaces for custom DL models.
    Child classes must be compatible with Hugging Face Transformers input and output classes.
    """

    @abstractmethod
    def to_transformers_input(self) -> Dict[str, Any]:
        """
        Converts the data model instance into a dictionary format compatible with
        Hugging Face Transformers model inputs.

        Returns:
            Dict[str, Any]: A dictionary representing the model's input.
        """
        pass