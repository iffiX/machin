from typing import Any
from dill import Pickler as DillPickler, loads as d_loads
from torch.multiprocessing import set_sharing_strategy
from torch.multiprocessing.reductions import reduce_event, reduce_storage, reduce_tensor
import io
import copyreg
import torch as t

# strategy "file_descriptor" will not work if sender
# process has been terminated before receiver process receives the tensor
# because the receiver needs to connect to sender to get FDs
set_sharing_strategy("file_system")


def mark_static_module(module: Any):  # pragma: no cover
    """
    Some modules are **static**, which means they are stateless
    and will remain the same whether you import it in process A
    or process B.

    If your module contains reference to functions, objects
    or anything inside a CDLL (usually the reference is a
    pointer), it is not picklable by dill, and will cause
    nasty errors, however, by marking this module as "Static",
    dill will recognize this module as a builtin module and
    not saving the states of this module, dill will only save
    a reference to it in this situation.

    Args:
        module: Some module which imports CDLLs by hand and
            not using pybind11.
    """
    del module.__file__


def _rebuild_full(data):
    buffer = io.BytesIO(data)
    return t.load(buffer)


def _reduce_full(obj):
    # supports saving tensors, storage, etc.
    # will always save all data and not by reference.
    buffer = io.BytesIO()
    t.save(obj, buffer)
    return _rebuild_full, (buffer.getvalue(),)


class Pickler(DillPickler):
    """
    Note:
        Picklers shares ".dispatch" among instances, and owns
        "dispatch_table" per instance.

        The base Pickler (not dill, from builtin pickle library),
        will first look up the default dump method in ".dispatch", if
        no valid method is found, it will try to find a custom dump
        method in ".dispatch_table".
    """

    def __init__(self, file, recurse=False, copy_tensor=False):
        super().__init__(file, byref=False, recurse=recurse)
        self.dispatch_table = copyreg.dispatch_table.copy()
        if not copy_tensor:
            # register the reduction methods provided by pytorch
            # same as init_reductions() in
            # torch.multiprocessing.reductions

            # In this case, receiver processes must be created by "fork",
            # and _share_memory()/share_memory() must be invoked on all
            # tensors/modules.
            # Otherwise "cpu" tensors will probably get a serious exception,
            # because the receiver processes are only getting pointers.
            # "cuda" tensors should will be fine
            self.dispatch_table[t.cuda.Event] = reduce_event
            for typ in t._storage_classes:
                self.dispatch_table[typ] = reduce_storage
            for typ in t._tensor_classes:
                self.dispatch_table[typ] = reduce_tensor
            self.dispatch_table[t.Tensor] = reduce_tensor
            self.dispatch_table[t.nn.parameter.Parameter] = reduce_tensor

        else:
            self.dispatch_table[t.cuda.Event] = reduce_event
            for typ in t._storage_classes:
                self.dispatch_table[typ] = _reduce_full
            for typ in t._tensor_classes:
                self.dispatch_table[typ] = _reduce_full
            self.dispatch_table[t.Tensor] = _reduce_full
            self.dispatch_table[t.nn.parameter.Parameter] = _reduce_full


def dumps(obj, recurse=False, copy_tensor=True):
    """
    Convert objects to bytes. Works for cpu and gpu tensors.

    Warning:
        Till pytorch 1.5.0, there is a bug for referenced gpu tensors,
        which would require users to keep shared gpu tensors during
        the whole process life and not reassigning / deleting them,
        however, you may refill them with different values.

        See `here <https://github.com/pytorch/pytorch/issues/39541>`_

    Args:
        obj: Object to dump.
        recurse: Enable recursive dumping, enable this to dump local
            functions and lambdas.
        copy_tensor: Whether to dump tensors, storage as a full copy.
            If it is set to "False", then dumped tensors must either
            locate on GPUs or in shared memory.

    Returns:
        Bytes.
    """
    buffer = io.BytesIO()
    pickler = Pickler(buffer, recurse, copy_tensor)
    pickler.dump(obj)
    return buffer.getvalue()


loads = d_loads
