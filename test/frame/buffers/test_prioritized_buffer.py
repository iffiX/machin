from machin.frame.transition import Transition
from machin.frame.buffers import WeightTree, PrioritizedBuffer

import dill
import pytest
import numpy as np
import torch as t


class TestWeightTree(object):
    WEIGHT_TREE_SIZE = 5
    WEIGHT_TREE_BASE = [1, 1, 1, 1, 3]
    WEIGHT_TREE_WEIGHTS = [1, 1, 1, 1, 3, 0, 0, 0,
                           2,    2,    3,    0,
                           4,          3,
                           7]

    @pytest.fixture(scope="class")
    def const_tree(self):
        tree = WeightTree(self.WEIGHT_TREE_SIZE)
        tree.update_all_leaves(self.WEIGHT_TREE_BASE)
        return tree

    ########################################################################
    # Test for WeightTree.get_weight_sum
    ########################################################################
    def test_get_weight_sum(self, const_tree):
        assert const_tree.get_weight_sum() == self.WEIGHT_TREE_WEIGHTS[-1]

    ########################################################################
    # Test for WeightTree.get_leaf_max
    ########################################################################
    def test_get_leaf_max(self, const_tree):
        assert const_tree.get_leaf_max() == max(self.WEIGHT_TREE_BASE)

    ########################################################################
    # Test for WeightTree.get_leaf_all_weights
    ########################################################################
    def test_get_leaf_all_weights(self, const_tree):
        assert np.all(const_tree.get_leaf_all_weights() ==
                      self.WEIGHT_TREE_BASE)

    ########################################################################
    # Test for WeightTree.get_leaf_weight
    ########################################################################
    param_get_leaf_weight = [
        (1, 1, None, None),
        ([1, 2], [1, 1], None, None),
        (np.array([0, 1, 2, 3, 4]), WEIGHT_TREE_BASE, None, None),
        (7, None, ValueError, "Index has elements out of boundary"),
        (-1, None, ValueError, "Index has elements out of boundary")
    ]

    @pytest.mark.parametrize("index,weight,exception,match",
                             param_get_leaf_weight)
    def test_get_leaf_weight(self, const_tree, index, weight, exception, match):
        if exception is not None:
            with pytest.raises(exception, match=match):
                const_tree.get_leaf_weight(index)
        else:
            assert np.all(const_tree.get_leaf_weight(index) ==
                          weight)

    ########################################################################
    # Test for WeightTree.find_leaf_index
    ########################################################################
    param_find_leaf_index = [
        (0.9, 0),
        (1.0, 0),
        (1.1, 1),
        ([-1.0, 0.9, 1.1, 2.0, 3.9, 4.2, 7.0, 10.0],
         [0, 0, 1, 1, 3, 4, 4, 4]),
        (np.array([-1.0, 0.9, 1.1, 2.0, 3.9, 4.2, 7.0, 10.0]),
         np.array([0, 0, 1, 1, 3, 4, 4, 4]))
    ]

    @pytest.mark.parametrize("weight,index", param_find_leaf_index)
    def test_find_leaf_index(self, const_tree, weight, index):
        assert np.all(const_tree.find_leaf_index(weight) == index)

    ########################################################################
    # Test for WeightTree.update_leaf
    ########################################################################
    param_update_leaf = [
        (2.0, 7, ValueError, "Index has elements out of boundary", None),
        (2.0, -1, ValueError, "Index has elements out of boundary", None),
        (2.0, 1, None, None, [1, 2, 1, 1, 3, 0, 0, 0,
                              3, 2, 3, 0,
                              5, 3,
                              8]),
    ]

    @pytest.mark.parametrize("weight,index,exception,match,new_weights",
                             param_update_leaf)
    def test_update_leaf(self, weight, index, exception, match, new_weights):
        tree = WeightTree(self.WEIGHT_TREE_SIZE)
        tree.update_all_leaves(self.WEIGHT_TREE_BASE)
        if exception is not None:
            with pytest.raises(exception, match=match):
                tree.update_leaf(weight, index)
        else:
            tree.update_leaf(2.0, 1)
            assert np.all(tree.weights == new_weights)

    ########################################################################
    # Test for WeightTree.update_leaf_batch
    ########################################################################
    param_update_leaf_batch = [
        ([1.0], [2, 3],
         ValueError, "Dimension of weights and indexes doesn't match", None),
        ([1.0, 2.0, 3.0], [3, WEIGHT_TREE_SIZE + 1, -1],
         ValueError, "Index has elements out of boundary", None),
        ([], [], None, None, WEIGHT_TREE_WEIGHTS),  # return directly
        ([2.0, 2.0], [1, 3], None, None,
         [1, 2, 1, 2, 3, 0, 0, 0,
          3,    3,    3,    0,
          6,          3,
          9])
    ]

    @pytest.mark.parametrize("weights,indexes,exception,match,new_weights",
                             param_update_leaf_batch)
    def test_update_leaf_batch(self, weights, indexes, exception, match,
                               new_weights):
        tree = WeightTree(self.WEIGHT_TREE_SIZE)
        tree.update_all_leaves(self.WEIGHT_TREE_BASE)
        if exception is not None:
            with pytest.raises(exception, match=match):
                tree.update_leaf_batch(weights, indexes)
        else:
            tree.update_leaf_batch(weights, indexes)
            assert np.all(tree.weights == new_weights)

    ########################################################################
    # Test for WeightTree.update_all_leaves
    ########################################################################
    def test_update_all_leaves(self):
        tree = WeightTree(self.WEIGHT_TREE_SIZE)
        tree.update_all_leaves(self.WEIGHT_TREE_BASE)
        assert np.all(tree.weights == self.WEIGHT_TREE_WEIGHTS)

        with pytest.raises(ValueError, match="must match tree size"):
            tree.update_all_leaves([1] * 3)

    ########################################################################
    # Test for WeightTree.print_weights
    ########################################################################
    def test_print_weights(self, const_tree):
        import sys, os
        # disable print output
        sys.stdout = open(os.devnull, 'w')
        const_tree.print_weights()
        # restore print output
        sys.stdout = sys.__stdout__


class TestPrioritizedBuffer(object):
    pass
