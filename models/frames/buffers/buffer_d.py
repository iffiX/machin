import numpy as np

from datetime import datetime as dt
from collections import OrderedDict

from .buffer import *
from utils.parallel.distributed import *


def _round_up(num):
    return int(np.ceil(num))


class DistributedBuffer(Buffer):
    def __init__(self, buffer_size, buffer_group=None, main_attributes=None, *_, **__):
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
        self.buffer_group = buffer_group if buffer_group is not None else group.WORLD
        self.buffer_group_head = 0
        self.collect_cache = OrderedDict()

    def all_size(self):
        """
        Returns:
            Total length of all buffers.
        """
        local_size = torch.tensor(len(self.buffer))
        all_reduce(local_size, group=self.buffer_group)
        return local_size.item()

    def request_batch(self, batch_size, sample_method):
        other_
        self.collect_batch()
        if isinstance(sample_method, str):
            if not hasattr(self, "sample_method_" + sample_method):
                raise RuntimeError("Cannot find specified sample method: {}".format(sample_method))
            sample_method = getattr(self, "sample_method_" + sample_method)

    def collect_batch(self):
        if get_rank(self.buffer_group) != -1:
            # if current process belongs to the buffer group
            p_num = get_world_size(self.buffer_group)

            # sample raw local batch from local buffer
            local_batch_size = _round_up(batch_size / p_num)
            local_batch, local_batch_size = sample_method(self.buffer, local_batch_size)

            # determine local data to send
            avail_size = torch.tensor(local_batch_size)
            all_reduce(avail_size, group=self.buffer_group)
            avail_size = avail_size.item()

            batch_size = min(avail_size, batch_size)
            # select_index is the final position of sample in gathered batches
            select_index = torch.arange(batch_size).split(_round_up(batch_size / p_num))
            select_index_pad = [t.tensor([], dtype=t.long) for _ in range(p_num - len(select_index))]
            select_index = list(select_index) + select_index_pad
            local_select_index = torch.tensor([])

            scatter(local_select_index, select_index, src=0, group=self.buffer_group)

            # exchange data
            local_batch_size = local_select_index.shape[0]
            local_batch = local_batch[:local_batch_size]
            batch_size, batch = self._gather_batches(local_batch)

            cur_time = int(dt.utcnow().timestamp())
            self.collect_cache[(cur_time, batch_size)] = batch

    def sample_batch(self, batch_size, concatenate=True, device=None,
                     sample_method="random_unique",
                     sample_keys=None,
                     additional_concat_keys=None, *_, **__):
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
            sample_keys: If sample_keys is specified, then only specified keys
                         of the transition object will be sampled.
            additional_concat_keys: additional custom keys needed to be concatenated, their
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
        if device is None:
            device = self.buffer_device
        if sample_keys is None:
            sample_keys = batch[0].keys()
        if additional_concat_keys is None:
            additional_concat_keys = []

        return batch_size, self.concatenate_batch(batch, batch_size, concatenate, device,
                                                  sample_keys, additional_concat_keys)

    def _gather_batches(self, batch):
        return {}

    def __reduce__(self):
        # for pickling
        return self.__class__, (self.buffer_size, self.buffer_device,
                                self.buffer_group, self.main_attrs)
