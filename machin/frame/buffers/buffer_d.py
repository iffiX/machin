from typing import Union, Dict, List, Any, Callable
from threading import RLock
from ..transition import Transition
from .buffer import Buffer
from machin.parallel.distributed import RpcGroup
import torch as t
import numpy as np
import itertools as it


def _round_up(num):
    return int(np.ceil(num))


class DistributedBuffer(Buffer):
    def __init__(self, buffer_name: str, group: RpcGroup, buffer_size: int, *_, **__):
        """
        Create a distributed replay buffer instance.

        To avoid issues caused by tensor device difference, all transition
        objects are stored in device "cpu".

        Distributed replay buffer constitutes of many local buffers held per
        process, transmissions between processes only happen during sampling.

        During sampling, the tensors in "state", "action" and "next_state"
        dictionaries, along with "reward", will be concatenated in dimension 0.
        any other custom keys specified in ``**kwargs`` will not be
        concatenated.

        .. seealso:: :class:`.Buffer`

        Note:
            Since ``append()`` operates on the local buffer, in order to
            append to the distributed buffer correctly, please make sure
            that your actor is also the local buffer holder, i.e. a member
            of the ``group``

        Args:
            buffer_size: Maximum local buffer size.
            group: Process group which holds this buffer.
            buffer_name: A unique name of your buffer.
        """
        super().__init__(buffer_size, "cpu")
        self.buffer_name = buffer_name
        self.group = group

        assert group.is_member()

        # register services, so that we may access other buffers
        _name = "/" + group.get_cur_name()
        self.group.register(buffer_name + _name + "/_size_service", self._size_service)
        self.group.register(
            buffer_name + _name + "/_clear_service", self._clear_service
        )
        self.group.register(
            buffer_name + _name + "/_sample_service", self._sample_service
        )
        self.wr_lock = RLock()

    def append(
        self,
        transition: Union[Transition, Dict],
        required_attrs=("state", "action", "next_state", "reward", "terminal"),
    ):
        # DOC INHERITED
        with self.wr_lock:
            super().append(transition, required_attrs=required_attrs)

    def clear(self):
        """
        Clear current local buffer.
        """
        with self.wr_lock:
            return super().clear()

    def all_clear(self):
        """
        Remove all entries from all local buffers.
        """
        future = [
            self.group.registered_async(self.buffer_name + "/" + m + "/_clear_service")
            for m in self.group.get_group_members()
        ]
        for fut in future:
            fut.wait()

    def size(self):
        """
        Returns:
            Length of current local buffer.
        """
        with self.wr_lock:
            return super().size()

    def all_size(self):
        """
        Returns:
            Total length of all buffers.
        """
        future = []
        count = 0
        for m in self.group.get_group_members():
            future.append(
                self.group.registered_async(
                    self.buffer_name + "/" + m + "/_size_service"
                )
            )
        for fut in future:
            count += fut.wait()
        return count

    def sample_batch(
        self,
        batch_size: int,
        concatenate: bool = True,
        device: Union[str, t.device] = None,
        sample_method: Union[Callable, str] = "random_unique",
        sample_attrs: List[str] = None,
        additional_concat_attrs: List[str] = None,
        *_,
        **__,
    ) -> Any:
        # DOC INHERITED
        p_num = self.group.size()
        local_batch_size = _round_up(batch_size / p_num)

        future = [
            self.group.registered_async(
                self.buffer_name + "/" + m + "/_sample_service",
                args=(local_batch_size, sample_method),
            )
            for m in self.group.get_group_members()
        ]

        results = [fut.wait() for fut in future]
        all_batch_size = sum([r[0] for r in results])
        all_batch = list(it.chain(*[r[1] for r in results]))

        if device is None:
            device = "cpu"
        if sample_attrs is None:
            sample_attrs = all_batch[0].keys()
        if additional_concat_attrs is None:
            additional_concat_attrs = []

        return (
            all_batch_size,
            Buffer.post_process_batch(
                all_batch, device, concatenate, sample_attrs, additional_concat_attrs
            ),
        )

    def _size_service(self):  # pragma: no cover
        return self.size()

    def _clear_service(self):  # pragma: no cover
        self.clear()

    def _sample_service(self, batch_size, sample_method):  # pragma: no cover
        if isinstance(sample_method, str):
            if not hasattr(self, "sample_method_" + sample_method):
                raise RuntimeError(
                    f"Cannot find specified sample method: {sample_method}"
                )
            sample_method = getattr(self, "sample_method_" + sample_method)

        # sample raw local batch from local buffer
        with self.wr_lock:
            local_batch_size, local_batch = sample_method(self.buffer, batch_size)

        return local_batch_size, local_batch
