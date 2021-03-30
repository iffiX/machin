from typing import Union, Dict, List, Any
from threading import RLock
from collections import OrderedDict
from ..transition import Transition
from .prioritized_buffer import PrioritizedBuffer
from machin.parallel.distributed import RpcGroup
import numpy as np
import torch as t


class DistributedPrioritizedBuffer(PrioritizedBuffer):
    def __init__(self, buffer_name: str, group: RpcGroup, buffer_size: int, *_, **__):
        """
        Create a distributed prioritized replay buffer instance.

        To avoid issues caused by tensor device difference, all transition
        objects are stored in device "cpu".

        Distributed prioritized replay buffer constitutes of many local buffers
        held per process, since it is very inefficient to maintain a weight
        tree across processes, each process holds a weight tree of records in
        its local buffer and a local buffer (same as ``DistributedBuffer``).

        The sampling process(es) will first use rpc to acquire the wr_lock,
        signalling "stop" to appending performed by actor processes,
        then perform a sum of all local weight trees, and finally perform
        sampling, after sampling and updating the importance weight,
        the lock will be released.


        During sampling, the tensors in "state", "action" and "next_state"
        dictionaries, along with "reward", will be concatenated in dimension 0.
        any other custom keys specified in ``**kwargs`` will not be
        concatenated.

        .. seealso:: :class:`PrioritizedBuffer`

        Note:
            :class:`DistributedPrioritizedBuffer` is not split into an
            accessor and an implementation, because we would like to operate
            on the buffer directly, when calling "size()" or "append()", to
            increase efficiency (since rpc layer is bypassed).

        Args:
            buffer_size: Maximum local buffer size.
            group: Process group which holds this buffer.
        """
        super().__init__(buffer_size, "cpu")
        self.buffer_name = buffer_name
        self.buffer_version_table = np.zeros([buffer_size], dtype=np.uint64)
        self.group = group

        assert group.is_member()

        # register services, so that we may access other buffers
        _name = "/" + group.get_cur_name()
        self.group.register(buffer_name + _name + "/_size_service", self._size_service)
        self.group.register(
            buffer_name + _name + "/_clear_service", self._clear_service
        )
        self.group.register(
            buffer_name + _name + "/_weight_sum_service", self._weight_sum_service
        )
        self.group.register(
            buffer_name + _name + "/_update_priority_service",
            self._update_priority_service,
        )
        self.group.register(
            buffer_name + _name + "/_sample_service", self._sample_service
        )
        self.wr_lock = RLock()

    def append(
        self,
        transition: Union[Transition, Dict],
        priority: Union[float, None] = None,
        required_attrs=("state", "action", "next_state", "reward", "terminal"),
    ):
        # DOC INHERITED
        with self.wr_lock:
            position = super(PrioritizedBuffer, self).append(transition, required_attrs)
            if priority is None:
                # the initialization method used in the original essay
                priority = self.wt_tree.get_leaf_max()
            self.wt_tree.update_leaf(self._normalize_priority(priority), position)
            # increase the version counter to mark it as tainted
            # later priority update will ignore this position
            self.buffer_version_table[position] += 1

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

    def clear(self):
        """
        Remove all entries from current local buffer.
        """
        with self.wr_lock:
            super().clear()
            # also clear the version table
            self.buffer_version_table.fill(0)

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

    def update_priority(self, priorities: np.ndarray, indexes: OrderedDict):
        # DOC INHERITED
        # update priority on all local buffers
        future = []
        offset = 0

        # indexes is an OrderedDict, key is sampled process name,
        # value is a tuple of an index np.ndarray and a version np.ndarray
        for m, sub in indexes.items():
            length = len(sub[0])
            future.append(
                self.group.registered_async(
                    self.buffer_name + "/" + m + "/_update_priority_service",
                    args=(priorities[offset : offset + length], sub[0], sub[1]),
                )
            )
            offset += length
        for fut in future:
            fut.wait()

    def sample_batch(
        self,
        batch_size: int,
        concatenate: bool = True,
        device: Union[str, t.device] = None,
        sample_attrs: List[str] = None,
        additional_concat_attrs: List[str] = None,
        *_,
        **__
    ) -> Any:
        if batch_size <= 0:
            return 0, None, None, None

        # calculate all weight sum
        future = [
            self.group.registered_async(
                self.buffer_name + "/" + m + "/_weight_sum_service"
            )
            for m in self.group.get_group_members()
        ]
        weights = [fut.wait() for fut in future]
        all_weight_sum = sum(weights) + 1e-6  # prevent all zero

        # determine the sampling size of local buffers, based on:
        # local_weight_sum / all_weight_sum
        ssize = np.ceil(np.array(weights) * batch_size / all_weight_sum)
        ssize = [int(ss) for ss in ssize]
        # collect samples and their priority
        future = [
            (
                m,
                self.group.registered_async(
                    self.buffer_name + "/" + m + "/_sample_service",
                    args=(ss, all_weight_sum),
                ),
            )
            for m, ss in zip(self.group.get_group_members(), ssize)
        ]

        all_batch_len = 0
        all_batch = []
        all_index = OrderedDict()
        all_is_weight = []
        for m, fut in future:
            batch_len, batch, index, version, is_weight = fut.wait()
            if batch_len == 0:
                continue
            all_batch_len += batch_len
            all_batch += batch
            all_is_weight.append(is_weight)
            # store them together to make API compatible with PrioritizedBuffer
            all_index[m] = (index, version)
        if all_batch_len == 0:
            return 0, None, None, None
        all_batch = PrioritizedBuffer.post_process_batch(
            all_batch, device, concatenate, sample_attrs, additional_concat_attrs
        )
        all_is_weight = np.concatenate(all_is_weight, axis=0)
        return all_batch_len, all_batch, all_index, all_is_weight

    def _size_service(self):  # pragma: no cover
        with self.wr_lock:
            return super().size()

    def _clear_service(self):  # pragma: no cover
        with self.wr_lock:
            super().clear()
            # also clear the version table
            self.buffer_version_table.fill(0)

    def _weight_sum_service(self):  # pragma: no cover
        with self.wr_lock:
            return self.wt_tree.get_weight_sum()

    def _update_priority_service(
        self, priorities, indexes, versions
    ):  # pragma: no cover
        with self.wr_lock:
            # compare original entry versions to the current version table
            is_same = self.buffer_version_table[indexes] == versions
            # select unchanged entries
            priorities = priorities[is_same]
            indexes = indexes[is_same]
            super().update_priority(priorities, indexes)

    def _sample_service(self, batch_size, all_weight_sum):  # pragma: no cover
        # the local batch size
        with self.wr_lock:
            if batch_size <= 0 or len(self.buffer) == 0:
                return 0, None, None, None, None

            wt_tree = self.wt_tree

            segment_length = wt_tree.get_weight_sum() / batch_size
            rand_priority = np.random.uniform(size=batch_size) * segment_length
            rand_priority += np.arange(batch_size, dtype=np.float) * segment_length
            rand_priority = np.clip(
                rand_priority, 0, max(wt_tree.get_weight_sum() - 1e-6, 0)
            )
            index = wt_tree.find_leaf_index(rand_priority)
            version = self.buffer_version_table[index]

            batch = [self.buffer[idx] for idx in index]
            priority = wt_tree.get_leaf_weight(index)

            # calculate importance sampling weight
            sample_probability = priority / all_weight_sum
            is_weight = np.power(len(self.buffer) * sample_probability, -self.curr_beta)
            is_weight /= is_weight.max()
            self.curr_beta = np.min(
                [1.0, self.curr_beta + self.beta_increment_per_sampling]
            )
            return len(batch), batch, index, version, is_weight
