from typing import Any
from torch.multiprocessing.reductions import ForkingPickler
import dill
import torch as t


def mark_static_module(module: Any):
    """
    Some modules are **static**, which means they are stateless
    and will remain the same whether you import it in process A
    or process B.

    If your module contains reference to functions, objects
    or anything inside a CDLL (usually the reference is a
    pointer), it is not pickable by dill, and will cause
    nasty errors, however, by marking this module as "Static",
    dill will recognize this module as a builtin module and
    not saving the states of this module, dill will only save
    a reference to it in this situation.

    Args:
        module: Some module which imports CDLLs by hand and
            not using pybind11.
    """
    del module.__file__


def dump_tensor(tensor: t.Tensor, reduce_as_reference=False):
    """
    Convert tensor to bytes. Works for cpu and gpu tensors.

    Warning:
        Till pytorch 1.5.0, there is a bug for referenced gpu tensors,
        which would require users to keep shared gpu tensors during
        the whole process life and not reassigning / deleting them,
        however, you may refill them with different values.

        See `here <https://github.com/pytorch/pytorch/issues/39541>`_

    Args:
        tensor: Some tensor.
        reduce_as_reference: Whether to dump the tensor as a reference,
            useful in same-node cross-process tensor transmission.

    Returns:
        Bytes.
    """
    if reduce_as_reference:
        # Only works on the same node, where tensor transmission
        # is needed between processes.
        # Since the registered reduction functions also return
        # constructor function (in torch/multiprocessing/reductions.py)
        # we can use dill to directly loads tensors.
        return ForkingPickler.dumps(tensor)
    else:
        # Pytorch implemented __reduce__ in the _StorageBase class
        # Storage is the base of all types of storage, and the
        # container of all types of tensors.
        # Internally it uses torch.save to convert the tensor to
        # raw bytes.
        tensor.share_memory_()
        return dill.dumps(tensor)
