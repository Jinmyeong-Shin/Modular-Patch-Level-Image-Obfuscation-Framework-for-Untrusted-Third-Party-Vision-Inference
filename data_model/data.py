from abc import ABC, abstractmethod

from typing import Any, Dict
from typing import Type, TypeVar

from dataclasses import asdict

class BaseModel(ABC):
    """
    Abstract Base Class for data models used as input/output interfaces for models.
    It provides default serialization/deserialization methods that work well
    with dataclasses.
    """

    _T = TypeVar('_T', bound='BaseModel')

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the model to a dictionary representation.

        The default implementation uses `dataclasses.asdict`, which is suitable
        for subclasses that are dataclasses. For other types of subclasses,
        this method may need to be overridden. For example, for dictionary-like
        objects, the implementation could be `return dict(self)`.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls: Type[_T], data: Dict[str, Any]) -> _T:
        """
        Creates a model instance from a dictionary.
        """
        return cls(**data)

    def __repr__(self) -> str:
        """
        Returns a string representation of the model.
        """
        return str(self.to_dict())
    
    