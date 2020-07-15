from typing import Union, Dict, List, Any
from threading import Lock
from collections import OrderedDict
from ..transition import Transition
from .prioritized_buffer import PrioritizedBuffer
from machin.parallel.distributed import RpcGroup
import numpy as np
import torch as t

from machin.utils.logging import default_logger
class DistributedPrioritizedBuffer(PrioritizedBuffer):
    def __init__(self, buffer_size: int, buffer_group: RpcGroup,
                 buffer_name: str = "dist_p_buffer", timeout: float = 1,
                 *_, **__):
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
            Since ``append()`` operates on the local buffer, in order to
            append to the distributed buffer correctly, please make sure
            that your actor is also the local buffer holder, i.e. a member
            of the ``buffer_group``

        Args:
            buffer_size: Maximum local buffer size.
            buffer_group: Process group which holds this buffer.
        """
        super(DistributedPrioritizedBuffer, self) \
            .__init__(buffer_size, "cpu")
        self.buffer_name = buffer_name
        self.buffer_group = buffer_group
        self.buffer_group.rpc_pair(buffer_name, self)
        self.timeout = timeout
        self.wr_lock = Lock()

    def append(self,
               transition: Union[Transition, Dict],
               priority: Union[float, None] = None,
               required_attrs=("state", "action", "next_state",
                               "reward", "terminal")):
        # DOC INHERITED
        with self.wr_lock:
            super(DistributedPrioritizedBuffer, self).append(
                transition, priority, required_attrs
            )

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
                m, self.buffer_name, self.size, timeout=self.timeout
            ))
        for fut in future:
            count += fut.wait()
        return count

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

    def update_priority(self,
                        priorities: np.ndarray,
                        indexes: OrderedDict):
        # DOC INHERITED
        # update priority on all local buffers
        future = []
        offset = 0
        for m, sub_idx in indexes.items():
            length = len(sub_idx)
            future.append(self.buffer_group.rpc_paired_class_async(
                m, self.buffer_name, self._rpc_reply_update_priority,
                args=(priorities[offset: offset + length], sub_idx),
                timeout=self.timeout
            ))
            offset += length
        for fut in future:
            fut.wait()

        # release the wr-lock on all local buffers
        future = [
            self.buffer_group.rpc_paired_class_async(
                m, self.buffer_name, self._rpc_reply_lock,
                args=(False,), timeout=self.timeout
            )
            for m in self.buffer_group.get_group_members()
        ]
        for fut in future:
            fut.wait()

    def sample_batch(self,
                     batch_size: int,
                     concatenate: bool = True,
                     device: Union[str, t.device] = None,
                     sample_attrs: List[str] = None,
                     additional_concat_attrs: List[str] = None,
                     *_, **__) -> Any:
        if batch_size <= 0:
            return 0, None, None, None

        # acquire the wr-lock on all local buffers.
        future = [
            self.buffer_group.rpc_paired_class_async(
                m, self.buffer_name, self._rpc_reply_lock,
                args=(True,), timeout=self.timeout
            )
            for m in self.buffer_group.get_group_members()
        ]
        for fut in future:
            fut.wait()

        # calculate all weight sum
        future = [
            self.buffer_group.rpc_paired_class_async(
                m, self.buffer_name, self._rpc_reply_weight_sum,
                timeout=self.timeout
            )
            for m in self.buffer_group.get_group_members()
        ]
        weights = [fut.wait() for fut in future]
        all_weight_sum = sum(weights) + 1e-6  # prevent all zero

        # determine the sampling size of local buffers, based on:
        # local_weight_sum / all_weight_sum
        ssize = np.ceil(np.array(weights) * batch_size / all_weight_sum)
        ssize = [int(ss) for ss in ssize]
        # collect samples and their priority
        future = [
            (m, 
             self.buffer_group.rpc_paired_class_async(
                 m, self.buffer_name, self._rpc_reply_sample,
                 args=(ss, all_weight_sum), timeout=self.timeout
             ))
            for m, ss in zip(self.buffer_group.get_group_members(), ssize)
        ]

        all_batch_len = 0
        all_batch = []
        all_index = OrderedDict()
        all_is_weight = []
        for m, fut in future:
            batch_len, batch, index, is_weight = fut.wait()
            if batch_len == 0:
                continue
            all_batch_len += batch_len
            all_batch += batch
            all_is_weight.append(is_weight)
            all_index[m] = index
        if all_batch_len == 0:
            return 0, None, None, None
        all_batch = self.post_process_batch(all_batch, device, concatenate,
                                            sample_attrs,
                                            additional_concat_attrs)
        all_is_weight = np.concatenate(all_is_weight, axis=0)
        return all_batch_len, all_batch, all_index, all_is_weight

    def _rpc_reply_clear(self):  # pragma: no cover
        with self.wr_lock:
            super(DistributedPrioritizedBuffer, self).clear()

    def _rpc_reply_weight_sum(self):  # pragma: no cover
        return self.wt_tree.get_weight_sum()

    def _rpc_reply_update_priority(self,
                                   priorities, indexes):  # pragma: no cover
        super(DistributedPrioritizedBuffer, self).update_priority(priorities,
                                                                  indexes)

    def _rpc_reply_sample(self, batch_size, all_weight_sum):  # pragma: no cover
        # the local batch size
        if batch_size <= 0 or self.size() == 0:
            return 0, None, None, None

        segment_length = self.wt_tree.get_weight_sum() / batch_size
        rand_priority = np.random.uniform(size=batch_size) * segment_length
        rand_priority += np.arange(batch_size, dtype=np.float) * segment_length
        rand_priority = np.clip(rand_priority, 0,
                                max(self.wt_tree.get_weight_sum() - 1e-6, 0))
        index = self.wt_tree.find_leaf_index(rand_priority)

        batch = [self.buffer[idx] for idx in index]
        priority = self.wt_tree.get_leaf_weight(index)

        # calculate importance sampling weight
        sample_probability = priority / all_weight_sum
        is_weight = np.power(len(self.buffer) * sample_probability, -self.beta)
        is_weight /= is_weight.max()
        self.curr_beta = np.min(
            [1., self.curr_beta + self.beta_increment_per_sampling]
        )
        return len(batch), batch, index, is_weight

    def _rpc_reply_lock(self, acquire):  # pragma: no cover
        if acquire:
            self.wr_lock.acquire()
        else:
            self.wr_lock.release()

    def __reduce__(self):
        # create a handle
        return DistributedPrioritizedBuffer, \
               (self.buffer_size, self.buffer_group,
                self.buffer_name, self.timeout)
