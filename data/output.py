from collections import OrderedDict
from functools import partial

from dataclasses import  fields, is_dataclass
from typing import Any

from .utils import _model_output_flatten, _model_output_unflatten

class Output(OrderedDict):
    """
    Base class for all outputs as dataclass. 
    Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. 
    Otherwise behaves like a regular python dictionary.

    <Tip warning={true}>

    You can't unpack a `Output` directly. 
    Use the [`~utils.Output.to_tuple`] method to convert it to a tuple before.

    </Tip>

    This code references the Hugging Face's transformers package.
    """

    def __init_subclass__(cls) -> None:
        """
        Registers the `Output` subclass with PyTorch's pytree utility.

        This allows instances of the subclass to be seamlessly used as inputs and
        outputs of PyTorch functions that operate on nested structures of tensors,
        such as in `torch.distributed` or `torch.fx`.
        """
        from torch.utils._pytree import register_pytree_node

        register_pytree_node(
            cls,
            _model_output_flatten,
            partial(_model_output_unflatten, output_type=cls),
            serialized_type_name=f"{cls.__module__}.{cls.__name__}",
        )
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        is_modeloutput_subclass = self.__class__ != Output

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f'{self.__module__}.{self.__class__.__name__} is not a dataclass.'
                ' This is a subclass of Output and so must use the @dataclass decorator.'
            )
        
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value) # Don't call self.__setitem__ to avoid recursion errors
        super().__setattr__(name, value) # Don't call self.__setattr__ to avoid recursion errors

    def __setitem__(self, key, value):
        super().__setitem__(key, value) # Will raise a KeyException if needed
        super().__setattr__(key, value) # Don't call self.__setattr__ to avoid recursion errors

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> tuple[Any]:
        """
        Converts the `Output` into a tuple of its values.

        This method allows the `Output` object to be unpacked like a regular
        tuple. The order of the elements in the resulting tuple corresponds to the
        order of fields in the dataclass definition.

        Returns:
            A tuple containing all the values of the `Output` instance.
        """
        return tuple(self[k] for k in self.keys())