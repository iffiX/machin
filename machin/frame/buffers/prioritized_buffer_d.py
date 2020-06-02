from random import choice
from typing import Union, Dict
from threading import Lock
from ..transition import Transition
from .prioritized_buffer import PrioritizedBuffer
from machin.utils.parallel.distributed import RpcGroup


class DistributedPrioritizedBuffer(PrioritizedBuffer):
    def __init__(self, buffer_size: int, buffer_group: RpcGroup, *_, **__):
        """
        Create a distributed replay buffer instance.

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
        any other custom keys specified in **kwargs will not be concatenated.

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

    def clear(self):
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

    def update_priority(self, priorities, indexes):
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

    def sample_batch(self, batch_size, concatenate=True, device=None,
                     sample_attrs=None, additional_concat_attrs=None, *_, **__):
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
