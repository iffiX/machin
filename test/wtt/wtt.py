import numpy as np
from typing import Union, List
from utils.helper_classes import Timer

class PrtReplayBuffer:
    def __init__(self, buffer_size):
        self.height = 0
        self.sizes = None
        self.offsets = None
        self.weights = None
        self.buffer_size = buffer_size
        self._wt_init()

    def _wt_init(self):
        """
        Initialize weight tree
        Note: weight tree is stored as a flattened, full binary tree in a numpy array.
              The lowest level of leaves comes first, the highest root node is stored
              at last.
        """
        # depth: level of nodes
        self.depth = int(np.ceil(np.log2(self.buffer_size))) + 1
        level_sizes_log = np.arange(self.depth - 1, -1, -1)
        self.sizes = np.power(2, level_sizes_log)
        self.offsets = np.concatenate(([0], np.cumsum(self.sizes)))
        self.weights = np.zeros([self.offsets[-1]], dtype=np.float)

    def _wt_build(self):
        """
        Build weight tree from leaves
        """
        for i in range(self.depth - 1):
            offset = self.offsets[i]
            level_size = self.sizes[i]
            # data are interleaved, therefore must be -1,2 and not 2,-1
            sum = self.weights[offset: offset + level_size].reshape(-1, 2).sum(axis=1)

            offset += level_size
            next_level_size = self.sizes[i + 1]
            self.weights[offset: offset + next_level_size] = sum

    def _wt_get_weight_sum(self):
        return self.weights[-1]

    def _wt_get_leaf_weights(self):
        return self.weights[range(self.sizes[0])]

    def _wt_find_leaf_index(self, weight: float):
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
        index = np.clip(index, 0, self.buffer_size - 1)

        if index.shape[0] == 1:
            return int(index)
        else:
            return index

    def _wt_update_leaf(self, weight: float, index: int):
        """
        Update weight tree leaf.
        """
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

    def _wt_update_leaf_batch(self,
                              weights: Union[List[float], np.ndarray],
                              indexes: Union[List[int], np.ndarray]):
        """
        Update weight tree leaves in batch.
        """
        if len(weights) != len(indexes):
            raise RuntimeError("Dimension of weights and indexes doesn't match!")
        if np.any(np.array(indexes) >= self.buffer_size):
            raise RuntimeError("Index has elements above buffer size boundary!")
        if len(weights) == 0:
            return

        weights = np.array(weights)
        indexes = np.array(indexes)
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

    def _wt_update_all_leaves(self, weights: Union[List[float], np.ndarray]):
        if len(weights) != self.buffer_size:
            raise RuntimeError("Weights size must match buffer size!")
        self.weights[0: len(weights)] = np.array(weights)
        self._wt_build()

    def _wt_print_weights(self, precision=2):
        fmt = "{{:.{}f}}".format(precision)
        for i in range(self.depth):
            offset, size = self.offsets[i], self.sizes[i]
            weights = [fmt.format(self.weights[j]) for j in range(offset, offset + size)]
            print(weights)


if __name__ == "__main__":
    x = PrtReplayBuffer(10)
    t = Timer()
    t.begin()
    x._wt_update_all_leaves([1 for _ in range(10)])
    x._wt_print_weights()

    new_weights = [2, 2, 2]
    indexes = [0, 5, 9]
    x._wt_update_leaf_batch(new_weights, indexes)
    x._wt_print_weights()

    x._wt_update_leaf(0, 5)
    x._wt_print_weights()

    x._wt_update_leaf(6, 10)
    x._wt_print_weights()

    print(np.cumsum(x._wt_get_leaf_weights()))
    print(x._wt_find_leaf_index(3))
    print(x._wt_find_leaf_index(np.array([0, 3, 9, 11, 17, 20])))


