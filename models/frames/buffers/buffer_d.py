import torch as t
import numpy as np
import itertools as it

from .buffer import Buffer, Transition, Union, Dict
from utils.parallel.distributed import Group
from threading import Lock


def _round_up(num):
    return int(np.ceil(num))


class DistributedBuffer(Buffer):
    def __init__(self, buffer_size, buffer_group: Group, main_attributes=None, *_, **__):
        """
        Create a distributed replay buffer instance.

        To avoid issues caused by tensor device difference, all transition
        objects are stored in device "cpu".

        Distributed replay buffer constitutes of many local buffers held per
        process, transmissions between processes only happen during sampling.

        Replay buffer stores a series of transition objects and functions
        as a ring buffer. The value of "state", "action", and "next_state"
        key must be a dictionary of tensors, the key of these tensors will
        be passed to your actor network and critics network as keyword
        arguments. You may store any additional info you need in the
        transition object, Values of "reward" and other keys
        will be passed to the reward function in DDPG/PPO/....

        During sampling, the tensors in "state", "action" and "next_state"
        dictionaries, along with "reward", will be concatenated in dimension 0.
        any other custom keys specified in **kwargs will not be concatenated.

        Note:
            You should not store any tensor inside **kwargs as they will not be
            moved to the sample output device.

        Args:
            buffer_size: Maximum local buffer size.
            buffer_device: Device where local buffer is stored.
            main_attributes: Major attributes where you can store tensors.
        """
        super(DistributedBuffer, self).__init__(buffer_size, "cpu", main_attributes)
        self.buffer_group = buffer_group
        self.buffer_group.rpc_register_paired(self.__class__)
        self.wr_lock = Lock()

    def all_size(self):
        """
        Returns:
            Total length of all buffers.
        """
        local_size = t.tensor(len(self.buffer), dtype=t.long)
        self.buffer_group.all_reduce(local_size)
        return local_size.item()

    def append(self, transition: Union[Transition, Dict],
               required_attrs=("state", "action", "next_state", "reward", "terminal")):
        self.wr_lock.acquire()
        super(DistributedBuffer, self).append(transition, required_attrs)
        self.wr_lock.release()

    def _request_batch(self, batch_size, sample_method):
        # TODO: add timeout
        future = [
            self.buffer_group.rpc_paired_class_async(
                p, self._reply_batch, self.__class__, args=(batch_size, sample_method)
            )
            for p in self._select_workers(batch_size)
        ]
        results = [fut.wait() for fut in future]
        all_batch_size = sum([r[0] for r in results])
        all_batch = list(it.chain([r[1] for r in results]))
        return all_batch_size, all_batch

    def _reply_batch(self, batch_size, sample_method):
        if isinstance(sample_method, str):
            if not hasattr(self, "sample_method_" + sample_method):
                raise RuntimeError("Cannot find specified sample method: {}".format(sample_method))
            sample_method = getattr(self, "sample_method_" + sample_method)

        p_num = self.buffer_group.size()

        # sample raw local batch from local buffer
        local_batch_size = _round_up(batch_size / p_num)
        self.wr_lock.acquire()
        local_batch, local_batch_size = sample_method(self.buffer, local_batch_size)
        self.wr_lock.release()

        return local_batch_size, local_batch

    def sample_batch(self, batch_size, concatenate=True, device=None,
                     sample_method="random_unique",
                     sample_attrs=None,
                     additional_concat_attrs=None, *_, **__):
        """
        Sample a random batch from replay buffer.

        Args:
            batch_size: Maximum size of the sample.
            sample_method: Sample method, could be a string of "random", "random_unique",
                           "all", or a function(list, batch_size) -> result, result_size.
            concatenate: Whether concatenate state, action and next_state in dimension 0.
                         If True, return a tensor with dim[0] = batch_size.
                         If False, return a list of tensors with dim[0] = 1.
            device:      Device to copy to.
            sample_attrs: If sample_keys is specified, then only specified keys
                         of the transition object will be sampled.
            additional_concat_attrs: additional custom keys needed to be concatenated, their
                                    value must be int, float or any numerical value, and must
                                    not be tensors.

        Returns:
            None if no batch is sampled.

            Or a tuple of sampled results, the tensors in "state", "action" and
            "next_state" dictionaries, along with "reward", will be concatenated
            in dimension 0 (if concatenate=True). If singular reward is float,
            it will be turned into a (1, 1) tensor, then concatenated. Any other
            custom keys will not be concatenated, just put together as lists.
        """
        batch_size, batch = self._request_batch(batch_size, sample_method)

        if device is None:
            device = self.buffer_device
        if sample_attrs is None:
            sample_attrs = batch[0].keys()
        if additional_concat_attrs is None:
            additional_concat_attrs = []

        return batch_size, self.post_process_batch(batch, device, concatenate,
                                                   sample_attrs, additional_concat_attrs)

    def _select_workers(self, num):
        workers = self.buffer_group.get_peer_ranks().copy()
        np.random.shuffle(workers)
        return workers[:num]

    def __reduce__(self):
        raise RuntimeError("Distributed buffer is not pickable, it is meant to be held per process!")
