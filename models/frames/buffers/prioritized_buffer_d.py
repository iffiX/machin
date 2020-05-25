from .buffer_d import Lock, Group
from .prioritized_buffer import *


class DistributedPrioritizedBuffer(PrioritizedBuffer):
    def __init__(self, buffer_size, buffer_group: Group, main_attributes=None, *_, **__):
        super(DistributedPrioritizedBuffer, self).__init__(buffer_size, "cpu", main_attributes)
        self.buffer_group = buffer_group
        self.buffer_group.rpc_register_paired(self.__class__)
        self.wr_lock = Lock()

    def append(self, transition: Union[Transition, Dict], priority: Union[float, None] = None,
               required_keys=("state", "action", "next_state", "reward", "terminal")):
        # TODO: batched append
        future = [self.buffer_group.rpc_paired_class_sync(
            p, self._reply_append, self.__class__,
            args=(transition, priority, required_keys))
            for p in self.buffer_group.get_peer_ranks()
        ]
        results = [fut.wait() for fut in future]
        if not all(results):
            failed = [p for p, status in zip(self.buffer_group.get_peer_ranks(), results) if not status]
            raise RuntimeError("Failed to perform append on process {}".format(failed))

    def clear(self):
        future = [self.buffer_group.rpc_paired_class_sync(
            p, self._reply_clear, self.__class__)
            for p in self.buffer_group.get_peer_ranks()
        ]
        results = [fut.wait() for fut in future]
        if not all(results):
            failed = [p for p, status in zip(self.buffer_group.get_peer_ranks(), results) if not status]
            raise RuntimeError("Failed to perform append on process {}".format(failed))

    def update_priority(self, priorities, indexes):
        future = [self.buffer_group.rpc_paired_class_sync(
            p, self._reply_update_priority, self.__class__,
            args=(priorities, indexes))
            for p in self.buffer_group.get_peer_ranks()
        ]
        results = [fut.wait() for fut in future]
        if not all(results):
            failed = [p for p, status in zip(self.buffer_group.get_peer_ranks(), results) if not status]
            raise RuntimeError("Failed to perform update priority on process {}".format(failed))

    def sample_batch(self, batch_size, concatenate=True, device=None,
                     sample_keys=None, additional_concat_keys=None, *_, **__):
        worker = np.min(np.abs(np.array(self.buffer_group.get_peer_ranks()) -
                               self.buffer_group.get_cur_rank()))
        return self.buffer_group.rpc_paired_class_sync(worker, self._reply_sample, self.__class__,
                                                       args=(batch_size, concatenate, device,
                                                             sample_keys, additional_concat_keys))

    def _reply_clear(self):
        self.wr_lock.acquire()
        super(DistributedPrioritizedBuffer, self).clear()
        self.wr_lock.release()
        return True

    def _reply_update_priority(self, priorities, indexes):
        self.wr_lock.acquire()
        super(DistributedPrioritizedBuffer, self).update_priority(priorities, indexes)
        self.wr_lock.release()
        return True

    def _reply_append(self, transition, priority, required_keys):
        self.wr_lock.acquire()
        super(DistributedPrioritizedBuffer, self).append(transition, priority, required_keys)
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
