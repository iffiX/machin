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
    def __init__(self, buffer_size: int, buffer_group: RpcGroup, *_, **__):
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
        """
        super(DistributedBuffer, self).__init__(buffer_size, "cpu")
        self.buffer_group = buffer_group
        self.buffer_group.rpc_register_paired(self.__class__, self)
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
                m, self._reply_clear, self.__class__
            )
            for m in self.buffer_group.get_group_members()
        ]
        results = [fut.wait() for fut in future]
        if not all(results):
            failed = [m for m, status
                      in zip(self.buffer_group.get_group_members(), results)
                      if not status]
            raise RuntimeError("Failed to perform clear on members {}"
                               .format(failed))

    def all_size(self):
        """
        Returns:
            Total length of all buffers.
        """
        future = []
        count = 0
        for m in self.buffer_group.get_group_members():
            future.append(self.buffer_group.rpc_paired_class_async(
                m, self.size, self.__class__
            ))
        for fut in future:
            count += fut.wait()
        return count

    def append(self, transition: Union[Transition, Dict],
               required_attrs=("state", "action", "next_state",
                               "reward", "terminal")):
        # DOC INHERITED
        self.wr_lock.acquire()
        super(DistributedBuffer, self).append(transition, required_attrs)
        self.wr_lock.release()

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
        # TODO: add timeout
        p_num = self.buffer_group.size()
        local_batch_size = _round_up(batch_size / p_num)

        future = [
            self.buffer_group.rpc_paired_class_async(
                w, self._reply_batch, self.__class__,
                args=(local_batch_size, batch_size, sample_method)
            )
            for w in self._select_workers(batch_size)
        ]

        results = [fut.wait() for fut in future]
        all_batch_size = sum([r[0] for r in results])
        all_batch = list(it.chain([r[1] for r in results]))
        return all_batch_size, all_batch

    def _reply_batch(self, batch_size, sample_method):
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

        self.wr_lock.acquire()
        local_batch_size, local_batch = sample_method(self.buffer,
                                                      batch_size)
        self.wr_lock.release()

        return local_batch_size, local_batch

    def _reply_clear(self, transition, priority, required_keys):
        self.wr_lock.acquire()
        super(DistributedBuffer, self).clear()
        self.wr_lock.release()
        return True

    def _select_workers(self, num: int):
        """
        Randomly select a sub group of workers.
        """
        workers = self.buffer_group.get_group_members().copy()
        np.random.shuffle(workers)
        return workers[:num]

    def __reduce__(self):
        raise RuntimeError("Distributed buffer is not pickable, "
                           "it is meant to be held per process!")
