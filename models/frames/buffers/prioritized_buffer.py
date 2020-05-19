import numpy as np
from typing import List
from .buffer import *


class WeightTree:
    def __init__(self, size):
        """
        Initialize a weight tree.
        Note: Weights must be positive.

        Note: Weight tree is stored as a flattened, full binary tree in a numpy array.
              The lowest level of leaves comes first, the highest root node is stored
              at last.
        Eg: Tree with weights: [[1, 2, 3, 4], [3, 7], [11]]
            will be stored as: [1, 2, 3, 4, 3, 7, 11]

        On i7-6700HQ (M: Million):
        90ms for building a tree with 10M elements.
        230ms for looking up 10M elements in a tree with 10M elements.
        20ms for 1M element batched update in a tree with 10M elements.
        240ms for 1M element single update in a tree with 10M elements.
        """
        self.size = size
        # depth: level of nodes
        self.max_leaf = -np.inf
        self.depth = int(np.ceil(np.log2(self.size))) + 1
        level_sizes_log = np.arange(self.depth - 1, -1, -1)
        self.sizes = np.power(2, level_sizes_log)
        self.offsets = np.concatenate(([0], np.cumsum(self.sizes)))
        self.weights = np.zeros([self.offsets[-1]], dtype=np.float)

    def build(self):
        """
        Build weight tree from leaves
        """
        self.max_leaf = self.get_leaf_all_weights().max()
        for i in range(self.depth - 1):
            offset = self.offsets[i]
            level_size = self.sizes[i]
            # data are interleaved, therefore must be -1,2 and not 2,-1
            sum = self.weights[offset: offset + level_size].reshape(-1, 2).sum(axis=1)

            offset += level_size
            next_level_size = self.sizes[i + 1]
            self.weights[offset: offset + next_level_size] = sum

    def get_weight_sum(self):
        return self.weights[-1]

    def get_leaf_max(self):
        return self.max_leaf

    def get_leaf_all_weights(self):
        return self.weights[range(self.sizes[0])]

    def get_leaf_weight(self, index: Union[int, List[int], np.ndarray]):
        if not isinstance(index, np.ndarray):
            index = np.array(index).reshape(-1)
        if np.any(np.array(index) >= self.size):
            raise RuntimeError("Index has elements above buffer size boundary!")
        return self.weights[index]

    def find_leaf_index(self, weight: Union[float, List[float], np.ndarray]):
        if not isinstance(weight, np.ndarray):
            weight = np.array(weight).reshape(-1)
        index = np.zeros([weight.shape[0]], dtype=np.long)

        # starting from the first child level of root
        for i in range(self.depth - 2, -1, -1):
            offset = self.offsets[i]
            left_wt = self.weights[offset + index * 2]
            # 0 for left and 1 for right
            select = weight > left_wt
            index = index * 2 + select
            weight = weight - left_wt * select
        index = np.clip(index, 0, self.size - 1)

        if index.shape[0] == 1:
            return int(index)
        else:
            return index

    def update_leaf(self, weight: float, index: int):
        """
        Update weight tree leaf.
        """
        self.max_leaf = max(weight, self.max_leaf)
        self.weights[index] = weight
        value = weight
        comp_value = self.weights[index ^ 0b1]

        for i in range(1, self.depth - 1):
            offset = self.offsets[i]
            index = index // 2
            global_index = index + offset
            comp_global_index = index ^ 0b1 + offset
            value = self.weights[global_index] = value + comp_value
            comp_value = self.weights[comp_global_index]

        self.weights[-1] = value + comp_value

    def update_leaf_batch(self,
                          weights: Union[List[float], np.ndarray],
                          indexes: Union[List[int], np.ndarray]):
        """
        Update weight tree leaves in batch.
        """
        if len(weights) != len(indexes):
            raise RuntimeError("Dimension of weights and indexes doesn't match!")
        if np.any(np.array(indexes) >= self.size):
            raise RuntimeError("Index has elements above buffer size boundary!")
        if len(weights) == 0:
            return

        weights = np.array(weights)
        indexes = np.array(indexes)

        self.max_leaf = max(np.max(weights), self.max_leaf)

        needs_update = indexes
        self.weights[indexes] = weights
        for i in range(1, self.depth):
            offset, prev_offset = self.offsets[i], self.offsets[i - 1]
            # O(n) = nlg(n)
            needs_update = np.unique(needs_update // 2)
            tmp = needs_update * 2
            self.weights[offset + needs_update] = \
                self.weights[prev_offset + tmp] + \
                self.weights[prev_offset + tmp + 1]

    def update_all_leaves(self, weights: Union[List[float], np.ndarray]):
        if len(weights) != self.size:
            raise RuntimeError("Weights size must match buffer size!")
        self.weights[0: len(weights)] = np.array(weights)
        self.build()

    def print_weights(self, precision=2):
        fmt = "{{:.{}f}}".format(precision)
        for i in range(self.depth):
            offset, size = self.offsets[i], self.sizes[i]
            weights = [fmt.format(self.weights[j]) for j in range(offset, offset + size)]
            print(weights)


class PrioritizedBuffer(Buffer):
    epsilon = 1e-2
    alpha = 0.6
    beta = 0.4
    curr_beta = beta
    beta_increment_per_sampling = 0.001

    def __init__(self, buffer_size, buffer_device="cpu", main_attributes=None):
        super(PrioritizedBuffer, self).__init__(buffer_size, buffer_device, main_attributes)
        self.wt_tree = WeightTree(buffer_size)

    def _normalize_priority(self, priority):
        return (np.abs(priority) + self.epsilon) ** self.alpha

    def append(self, transition: Union[Transition, Dict], priority: Union[float, None] = None):
        """
        Store a transition object to buffer.

        Args:
            transition: A transition object.
            priority: Priority of transition.
        """
        position = super(PrioritizedBuffer, self).append(transition)
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
        Sample a random batch from priortized buffer.

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
