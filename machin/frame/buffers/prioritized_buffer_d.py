from random import choice
from typing import Union, Dict, List, Tuple, Any
from threading import Lock
from ..transition import Transition
from .prioritized_buffer import PrioritizedBuffer
from machin.parallel.distributed import RpcGroup
import numpy as np
import torch as t


class DistributedPrioritizedBuffer(PrioritizedBuffer):
    def __init__(self, buffer_size: int, buffer_group: RpcGroup, *_, **__):
        """
        Create a distributed prioritized replay buffer instance.

        To avoid issues caused by tensor device difference, all transition
        objects are stored in device "cpu".

        Distributed prioritized replay buffer constitutes of many local buffers
        held per process, since it is very inefficient to maintain a weight
        tree across processes, each process holds a full copy of the weight
        tree and only a part of the buffer (same as ``DistributedBuffer``),
        transmissions between processes happen during appending, updating and
        sampling.

        During sampling, the tensors in "state", "action" and "next_state"
        dictionaries, along with "reward", will be concatenated in dimension 0.
        any other custom keys specified in ``**kwargs`` will not be
        concatenated.

        .. seealso:: :class:`PrioritizedBuffer`

        Args:
            buffer_size: Maximum local buffer size.
            buffer_group: Process group which holds this buffer.
        """
        super(DistributedPrioritizedBuffer, self) \
            .__init__(buffer_size, "cpu")
        self.buffer_group = buffer_group
        self.buffer_group.rpc_register_paired(self.__class__, self)
        self.wr_lock = Lock()

    def append(self,
               transition: Union[Transition, Dict],
               priority: Union[float, None] = None,
               required_attrs=("state", "action", "next_state",
                               "reward", "terminal")):
        # DOC INHERITED
        # TODO: batched append
        future = [
            self.buffer_group.rpc_paired_class_async(
                m, self._reply_append, self.__class__,
                args=(transition, priority, required_attrs)
            )
            for m in self.buffer_group.get_group_members()
        ]
        results = [fut.wait() for fut in future]
        if not all(results):
            failed = [m for m, status
                      in zip(self.buffer_group.get_group_members(), results)
                      if not status]
            raise RuntimeError("Failed to perform append on members {}"
                               .format(failed))

    def size(self):
        """
        Returns:
            Length of current local buffer.
        """
        return len(self.buffer)

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
            raise RuntimeError("Failed to perform append on members {}"
                               .format(failed))

    def update_priority(self, priorities: np.ndarray, indexes: np.ndarray):
        # DOC INHERITED
        future = [
            self.buffer_group.rpc_paired_class_async(
                m, self._reply_update_priority, self.__class__,
                args=(priorities, indexes)
            )
            for m in self.buffer_group.get_group_members()
        ]
        results = [fut.wait() for fut in future]
        if not all(results):
            failed = [m for m, status
                      in zip(self.buffer_group.get_group_members(), results)
                      if not status]
            raise RuntimeError("Failed to perform update priority on members {}"
                               .format(failed))

    def sample_batch(self,
                     batch_size: int,
                     concatenate: bool = True,
                     device: Union[str, t.device] = None,
                     sample_attrs: List[str] = None,
                     additional_concat_attrs: List[str] = None,
                     *_, **__) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
        # DOC INHERITED
        worker = choice(self.buffer_group.get_group_members())
        return self.buffer_group.rpc_paired_class_async(
            worker, self._reply_sample, self.__class__,
            args=(batch_size, concatenate, device,
                  sample_attrs, additional_concat_attrs)
        )

    def _reply_clear(self):
        self.wr_lock.acquire()
        super(DistributedPrioritizedBuffer, self).clear()
        self.wr_lock.release()
        return True

    def _reply_update_priority(self, priorities, indexes):
        self.wr_lock.acquire()
        super(DistributedPrioritizedBuffer, self).update_priority(priorities,
                                                                  indexes)
        self.wr_lock.release()
        return True

    def _reply_append(self, transition, priority, required_keys):
        self.wr_lock.acquire()
        super(DistributedPrioritizedBuffer, self).append(transition, priority,
                                                         required_keys)
        self.wr_lock.release()
        return True

    def _reply_sample(self, batch_size, concatenate, device,
                      sample_keys, additional_concat_keys):
        self.wr_lock.acquire()
        result = super(DistributedPrioritizedBuffer, self).sample_batch(
            batch_size, concatenate, device, sample_keys, additional_concat_keys
        )
        self.wr_lock.release()
        return result
