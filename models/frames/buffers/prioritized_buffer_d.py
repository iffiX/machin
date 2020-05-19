from .prioritized_buffer import *


class DistributedPrioritizedBuffer(Buffer):
    epsilon = 1e-2
    alpha = 0.6
    beta = 0.4
    curr_beta = beta
    beta_increment_per_sampling = 0.001

    def __init__(self, buffer_size, buffer_device="cpu", main_attributes=None):
        super(DistributedPrioritizedBuffer, self).__init__(buffer_size, buffer_device, main_attributes)
        self.wt_tree = WeightTree(buffer_size)

    def _normalize_priority(self, priority):
        return (np.abs(priority) + self.epsilon) ** self.alpha

    def append(self, transition: Union[Transition, Dict], priority: Union[float, None]=None):
        """
        Store a transition object to buffer.

        Args:
            transition: A transition object.
            priority: Priority of transition.
        """
        position = super(DistributedPrioritizedBuffer, self).append(transition)
        if priority is None:
            # the initialization method used in the original essay
            priority = self.wt_tree.get_leaf_max()
        self.wt_tree.update_leaf(self._normalize_priority(priority), position)

    def clear(self):
        self.buffer.clear()
        self.wt_tree = WeightTree(self.buffer_size)
        self.curr_beta = self.beta

    def update_priority(self, priorities, indexes):
        priorities = self._normalize_priority(priorities)
        self.wt_tree.update_leaf_batch(priorities, indexes)

    def sample_batch(self, batch_size, concatenate=True, device=None,
                     sample_keys=None, additional_concat_keys=None, *_, **__):
        """
        Sample a random batch from priortized replay buffer.

        Args:
            batch_size: Maximum size of the sample.
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
        segment_length = self.wt_tree.get_weight_sum() / batch_size
        self.curr_beta = np.min([1., self.curr_beta + self.beta_increment_per_sampling])

        rand_priority = np.random.uniform()
        rand_priority += np.arange(batch_size, dtype=np.float) * segment_length
        rand_priority = np.clip(rand_priority, 0, self.wt_tree.get_weight_sum() - 1e-6)
        index = self.wt_tree.find_leaf_index(rand_priority)

        batch = [self.buffer[idx] for idx in index]
        priority = self.wt_tree.get_leaf_weight(index)

        # calculate importance sampling weight
        sample_probability = priority / self.wt_tree.get_weight_sum()
        is_weight = np.power(len(self.buffer) * sample_probability, -self.beta)
        is_weight /= is_weight.max()

        # post processing
        if device is None:
            device = self.buffer_device
        if sample_keys is None:
            sample_keys = batch[0].keys()
        if additional_concat_keys is None:
            additional_concat_keys = []

        result = self.concatenate_batch(batch, batch_size, concatenate, device,
                                        sample_keys, additional_concat_keys)
        return batch_size, result, index, is_weight
