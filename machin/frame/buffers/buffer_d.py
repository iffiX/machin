from typing import Union, Dict, List, Any, Callable
from threading import Lock
from ..transition import Transition
from .buffer import Buffer
from machin.parallel.distributed import RpcGroup
import torch as t
import numpy as np
import itertools as it


def _round_up(num):
    return int(np.ceil(num))


class DistributedBuffer(Buffer):
    def __init__(self, buffer_size: int, buffer_group: RpcGroup,
                 buffer_name: str = "dist_buffer", timeout: float = 1,
                 *_, **__):
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

        Args:
            buffer_size: Maximum local buffer size.
            buffer_group: Process group which holds this buffer.
            buffer_name: A unique name of your buffer.
            timeout: Timeout value of rpc requests.
        """
        super(DistributedBuffer, self).__init__(buffer_size, "cpu")
        self.buffer_name = buffer_name
        self.buffer_group = buffer_group
        self.buffer_group.rpc_pair(buffer_name, self)
        self.timeout = timeout
        self.wr_lock = Lock()

    def size(self):
        """
        Returns:
            Length of current local buffer.
        """
        return len(self.buffer)

    def clear(self):
        # DOC INHERITED
        future = [
            self.buffer_group.rpc_paired_class_async(
                m, self.buffer_name, self._rpc_reply_clear,
                timeout=self.timeout
            )
            for m in self.buffer_group.get_group_members()
        ]
        for fut in future:
            fut.wait()

    def all_size(self):
        """
        Returns:
            Total length of all buffers.
        """
        future = []
        count = 0
        for m in self.buffer_group.get_group_members():
            future.append(self.buffer_group.rpc_paired_class_async(
                m, self.buffer_name, self.size, timeout=self.timeout
            ))
        for fut in future:
            count += fut.wait()
        return count

    def append(self, transition: Union[Transition, Dict],
               required_attrs=("state", "action", "next_state",
                               "reward", "terminal")):
        # DOC INHERITED
        with self.wr_lock:
            super(DistributedBuffer, self).append(transition, required_attrs)

    def sample_batch(self,
                     batch_size: int,
                     concatenate: bool = True,
                     device: Union[str, t.device] = None,
                     sample_method: Union[Callable, str] = "random_unique",
                     sample_attrs: List[str] = None,
                     additional_concat_attrs: List[str] = None,
                     *_, **__) -> Any:
        # DOC INHERITED
        batch_size, batch = self._request_batch(batch_size, sample_method)

        if device is None:
            device = self.buffer_device
        if sample_attrs is None:
            sample_attrs = batch[0].keys()
        if additional_concat_attrs is None:
            additional_concat_attrs = []

        return \
            batch_size, \
            self.post_process_batch(batch, device, concatenate,
                                    sample_attrs, additional_concat_attrs)

    def _request_batch(self, batch_size: int,
                       sample_method: Union[Callable, str]) -> Any:
        """
        Used by samplers. Samplers will split the query equally to all buffer
        holders, and receive their reply.

        Args:
            batch_size: Size of batch to sample.
            sample_method: Sample method.

        Returns:
            Sampled batch size, batch data.
        """
        p_num = self.buffer_group.size()
        local_batch_size = _round_up(batch_size / p_num)

        future = [
            self.buffer_group.rpc_paired_class_async(
                w, self.buffer_name, self._rpc_reply_batch,
                timeout=self.timeout,
                args=(local_batch_size, sample_method)
            )
            for w in self.buffer_group.get_group_members()
        ]

        results = [fut.wait() for fut in future]
        all_batch_size = sum([r[0] for r in results])
        all_batch = list(it.chain(*[r[1] for r in results]))
        return all_batch_size, all_batch

    def _rpc_reply_batch(self, batch_size, sample_method):  # pragma: no cover
        """
        Rpc function executed by buffer holders, will sample requested batch
        size.

        Args:
            batch_size: Size of batch to sample.
            sample_method: Sample method.

        Returns:
            Sampled batch size, batch data.
        """
        if isinstance(sample_method, str):
            if not hasattr(self, "sample_method_" + sample_method):
                raise RuntimeError("Cannot find specified sample method: {}"
                                   .format(sample_method))
            sample_method = getattr(self, "sample_method_" + sample_method)

        # sample raw local batch from local buffer
        with self.wr_lock:
            local_batch_size, local_batch = sample_method(self.buffer,
                                                          batch_size)

        return local_batch_size, local_batch

    def _rpc_reply_clear(self):  # pragma: no cover
        with self.wr_lock:
            super(DistributedBuffer, self).clear()

    def __reduce__(self):
        raise RuntimeError("Distributed buffer is not picklable, "
                           "it is meant to be held per process!")
