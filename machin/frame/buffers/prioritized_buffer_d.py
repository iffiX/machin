from typing import Union, Dict, List, Any
from threading import Lock
from collections import OrderedDict
from ..transition import Transition
from .prioritized_buffer import PrioritizedBuffer
from machin.parallel.distributed import RpcGroup, get_cur_name
import numpy as np
import torch as t


class DistributedPrioritizedBuffer:
    def __init__(self, buffer_name: str, group: RpcGroup):
        """
        an accessor to a distributed prioritized replay buffer instance.

        Args:
            buffer_name: A unique name of your buffer.
            group: Process group which holds this buffer.
        """
        self.buffer_name = buffer_name
        self.group = group

    def append(self,
               transition: Union[Transition, Dict],
               priority: Union[float, None] = None,
               required_attrs=("state", "action", "next_state",
                               "reward", "terminal"),
               buffer_process: str = None):
        """
        Store a transition object to buffer.

        Args:
            transition: A transition object.
            priority: Priority of transition.
            required_attrs: Required attributes. Could be an empty tuple if
                no attribute is required.
            buffer_process: The process holding the local buffer you would
                like to append to. You can omit this argument if current
                process is holding a local buffer.

        Raises:
            ``ValueError`` if transition object doesn't have required
            attributes in ``required_attrs`` or has different attributes
            compared to other transition objects stored in buffer.
        """
        if (get_cur_name() not in self.group.get_group_members() and
                buffer_process is None):
            raise ValueError('You must specify "buffer_process" because '
                             'current Process [{}] is not a member of '
                             'Group [{}]'
                             .format(get_cur_name(), self.group.group_name))
        return self.group.registered_sync(
            self.buffer_name + "/" + buffer_process + "/_append_service",
            args=(transition, priority, required_attrs)
        )

    def size(self, buffer_process: str = None):
        """
        Args:
            buffer_process: The process holding the local buffer you would
                like to query the size. You can omit this argument if current
                process is holding a local buffer.

        Returns:
            Length of current local buffer.
        """
        if (get_cur_name() not in self.group.get_group_members() and
                buffer_process is None):
            raise ValueError('You must specify "buffer_process" because '
                             'current Process [{}] is not a member of '
                             'Group [{}]'
                             .format(get_cur_name(), self.group.group_name))
        return self.group.registered_sync(
            self.buffer_name + "/" + buffer_process + "/_size_service"
        )

    def all_size(self):
        """
        Returns:
            Total length of all buffers.
        """
        future = []
        count = 0
        for m in self.group.get_group_members():
            future.append(self.group.registered_async(
                self.buffer_name + "/" + m + "/_size_service"
            ))
        for fut in future:
            count += fut.wait()
        return count

    def clear(self):
        """
        Remove all entries from all local buffers.
        """
        future = [
            self.group.registered_async(
                self.buffer_name + "/" + m + "/_clear_service"
            )
            for m in self.group.get_group_members()
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
            future.append(self.group.registered_async(
                self.buffer_name + "/" + m + "/_update_priority_service",
                args=(priorities[offset: offset + length], sub_idx)
            ))
            offset += length
        for fut in future:
            fut.wait()

        # release the wr-lock on all local buffers
        future = [
            self.group.registered_async(
                self.buffer_name + "/" + m + "/_lock_service",
                args=(False,)
            )
            for m in self.group.get_group_members()
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
            self.group.registered_async(
                self.buffer_name + "/" + m + "/_lock_service",
                args=(True,)
            )
            for m in self.group.get_group_members()
        ]
        for fut in future:
            fut.wait()

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
            (m,
             self.group.registered_async(
                 self.buffer_name + "/" + m + "/_sample_service",
                 args=(ss, all_weight_sum)
             ))
            for m, ss in zip(self.group.get_group_members(), ssize)
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
        all_batch = PrioritizedBuffer.post_process_batch(
            all_batch, device, concatenate, sample_attrs,
            additional_concat_attrs
        )
        all_is_weight = np.concatenate(all_is_weight, axis=0)
        return all_batch_len, all_batch, all_index, all_is_weight


class DistributedPrioritizedBufferImpl(PrioritizedBuffer):
    def __init__(self, buffer_name: str, group: RpcGroup,
                 buffer_size: int, timeout: float = 10,
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
            of the ``group``

        Args:
            buffer_size: Maximum local buffer size.
            group: Process group which holds this buffer.
        """
        super(DistributedPrioritizedBufferImpl, self) \
            .__init__(buffer_size, "cpu")
        self.buffer_name = buffer_name
        self.group = group
        if group.get_cur_name() == group.get_group_members()[0]:
            self.group.pair(buffer_name, self)

        _name = "/" + group.get_cur_name()

        self.group.register(buffer_name + _name + "/_size_service",
                            self._size_service)
        self.group.register(buffer_name + _name + "/_clear_service",
                            self._clear_service)
        self.group.register(buffer_name + _name + "/_append_service",
                            self._append_service)
        self.group.register(buffer_name + _name + "/_weight_sum_service",
                            self._weight_sum_service)
        self.group.register(buffer_name + _name + "/_update_priority_service",
                            self._update_priority_service)
        self.group.register(buffer_name + _name + "/_sample_service",
                            self._sample_service)
        self.group.register(buffer_name + _name + "/_lock_service",
                            self._lock_service)
        self.timeout = timeout
        self.wr_lock = Lock()

    def _size_service(self):  # pragma: no cover
        return len(self.buffer)

    def _clear_service(self):  # pragma: no cover
        with self.wr_lock:
            super(DistributedPrioritizedBufferImpl, self).clear()

    def _append_service(self,
                        transition: Union[Transition, Dict],
                        priority: Union[float, None] = None,
                        required_attrs=("state", "action", "next_state",
                                        "reward", "terminal")
                        ):  # pragma: no cover
        # DOC INHERITED
        with self.wr_lock:
            super(DistributedPrioritizedBufferImpl, self)\
                .append(transition, priority, required_attrs)

    def _weight_sum_service(self):  # pragma: no cover
        return self.wt_tree.get_weight_sum()

    def _update_priority_service(self,
                                 priorities, indexes):  # pragma: no cover
        super(DistributedPrioritizedBufferImpl, self)\
            .update_priority(priorities, indexes)

    def _sample_service(self, batch_size, all_weight_sum):  # pragma: no cover
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

    def _lock_service(self, acquire):  # pragma: no cover
        if acquire:
            self.wr_lock.acquire()
        else:
            self.wr_lock.release()

    def __reduce__(self):  # pragma: no cover
        return DistributedPrioritizedBuffer, (self.buffer_name, self.group)
