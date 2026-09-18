# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch


def update_tensor_inplace(dst: torch.Tensor, src: torch.Tensor):
    """Update ``dst`` in-place to match ``src`` shape and stride.

    If the underlying storage differs, copies data and deletes the
    old tensor reference.
    """
    assert dst.dtype == src.dtype, "Tensors must have the same dtype"

    # update tensor shape and stride
    dst.as_strided_(src.shape, src.stride())

    # If not the same underlying storage move tensor data
    if dst.data_ptr() != src.data_ptr():
        dst.copy_(src)
        del src


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_parameter(
    mod: torch.nn.Module, name: str, new: torch.Tensor | torch.nn.Parameter
) -> None:
    """Replace a module parameter with a new tensor, updating in-place
    when storage is compatible, otherwise re-registering the parameter.
    """
    old = getattr(mod, name)
    if (
        type(old) is type(new)
        and old.dtype == new.dtype
        and old.untyped_storage().nbytes() == new.untyped_storage().nbytes()
    ):
        # If we can just update in-place to avoid re-registering
        #   can be faster if the underlying storage is the same
        update_tensor_inplace(old, new)
    else:
        # Fallback re-register parameter, convert to Parameter if necessary
        # this not only ensures we don't register a tensor as a parameter, but
        # also ensures that all parameter subclasses get re-registered as
        # parameters for `torch.compile` compatibility
        if not isinstance(new, torch.nn.Parameter):
            new = torch.nn.Parameter(new, requires_grad=False)
        mod.register_parameter(name, torch.nn.Parameter(new, requires_grad=False))
