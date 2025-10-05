from typing import Iterable, Optional, Any, TYPE_CHECKING

import torch.utils._pytree as _torch_pytree

if TYPE_CHECKING:
    from .output import Output

def _model_output_flatten(output: "Output") -> tuple[list[Any], "_torch_pytree.Context"]:
    """
    Flattens a `Output` into a list of its values and a context.

    The context is the list of keys, which is needed to reconstruct the object.
    This function is registered with `torch.utils._pytree` to allow `Output`
    subclasses to be used in PyTorch functions that operate on nested structures
    of tensors.

    This code references the Hugging Face's transformers package.

    Args:
        output (`Output`): The `Output` instance to flatten.

    Returns:
        A tuple containing a list of the `Output`'s values and a list of its keys as context.
    """
    return list(output.values()), list(output.keys())

def _model_output_unflatten(
    values: Iterable[Any],
    context: "_torch_pytree.Context",
    output_type=None,
) -> "Output":
    """
    Unflattens a list of values and a context into a `Output` object.

    This function is the inverse of `_model_output_flatten` and is used by the
    `torch.utils._pytree` registry to reconstruct `Output` instances.

    This code references the Hugging Face's transformers package.

    Args:
        values (`Iterable[Any]`): An iterable of values (the flattened part).
        context (`_torch_pytree.Context`): The context (keys) from flattening.
        output_type (type): The specific `Output` subclass to create.
    """
    return output_type(**dict(zip(context, values)))