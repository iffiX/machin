from machin.frame.buffers import WeightTree, PrioritizedBuffer

import pytest
import numpy as np
import torch as t


class TestWeightTree(object):
    WEIGHT_TREE_SIZE = 5
    WEIGHT_TREE_BASE = [1, 1, 1, 1, 3]
    WEIGHT_TREE_WEIGHTS = [1, 1, 1, 1, 3, 0, 0, 0, 2, 2, 3, 0, 4, 3, 7]

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
        assert np.all(const_tree.get_leaf_all_weights() == self.WEIGHT_TREE_BASE)

    ########################################################################
    # Test for WeightTree.get_leaf_weight
    ########################################################################
    param_get_leaf_weight = [
        (1, 1, None, None),
        ([1, 2], [1, 1], None, None),
        (np.array([0, 1, 2, 3, 4]), WEIGHT_TREE_BASE, None, None),
        (7, None, ValueError, "Index has elements out of boundary"),
        (-1, None, ValueError, "Index has elements out of boundary"),
    ]

    @pytest.mark.parametrize("index,weight,exception,match", param_get_leaf_weight)
    def test_get_leaf_weight(self, const_tree, index, weight, exception, match):
        if exception is not None:
            with pytest.raises(exception, match=match):
                const_tree.get_leaf_weight(index)
        else:
            assert np.all(const_tree.get_leaf_weight(index) == weight)

    ########################################################################
    # Test for WeightTree.find_leaf_index
    ########################################################################
    param_find_leaf_index = [
        (0.9, 0),
        (1.0, 0),
        (1.1, 1),
        ([-1.0, 0.9, 1.1, 2.0, 3.9, 4.2, 7.0, 10.0], [0, 0, 1, 1, 3, 4, 4, 4]),
        (
            np.array([-1.0, 0.9, 1.1, 2.0, 3.9, 4.2, 7.0, 10.0]),
            np.array([0, 0, 1, 1, 3, 4, 4, 4]),
        ),
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
        (2.0, 1, None, None, [1, 2, 1, 1, 3, 0, 0, 0, 3, 2, 3, 0, 5, 3, 8]),
    ]

    @pytest.mark.parametrize(
        "weight,index,exception,match,new_weights", param_update_leaf
    )
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
        (
            [1.0],
            [2, 3],
            ValueError,
            "Dimension of weights and indexes doesn't match",
            None,
        ),
        (
            [1.0, 2.0, 3.0],
            [3, WEIGHT_TREE_SIZE + 1, -1],
            ValueError,
            "Index has elements out of boundary",
            None,
        ),
        ([], [], None, None, WEIGHT_TREE_WEIGHTS),  # return directly
        ([2.0, 2.0], [1, 3], None, None, [1, 2, 1, 2, 3, 0, 0, 0, 3, 3, 3, 0, 6, 3, 9]),
    ]

    @pytest.mark.parametrize(
        "weights,indexes,exception,match,new_weights", param_update_leaf_batch
    )
    def test_update_leaf_batch(self, weights, indexes, exception, match, new_weights):
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
        import sys
        import os

        # disable print output
        sys.stdout = open(os.devnull, "w")
        const_tree.print_weights()
        # restore print output
        sys.stdout = sys.__stdout__


class TestPrioritizedBuffer(object):
    ########################################################################
    # Test for PrioritizedBuffer.append
    ########################################################################
    param_test_append = [
        (
            [
                {
                    "state": {"state_1": t.zeros([1, 2])},
                    "action": {"action_1": t.zeros([1, 3])},
                    "next_state": {"next_state_1": t.zeros([1, 2])},
                    "reward": 1,
                    "terminal": True,
                    "some_custom_attr": None,
                }
            ],
            1,
            ((1 + 1e-2) ** 0.6, 0, 0, 0, 0),
            {
                "buffer_size": 5,
                "buffer_device": "cpu",
                "epsilon": 1e-2,
                "alpha": 0.6,
                "beta": 0.4,
                "beta_increment_per_sampling": 1e-3,
            },
        ),
        (
            [
                {
                    "state": {"state_1": t.zeros([1, 2])},
                    "action": {"action_1": t.zeros([1, 3])},
                    "next_state": {"next_state_1": t.zeros([1, 2])},
                    "reward": 1,
                    "terminal": True,
                    "some_custom_attr": None,
                }
            ],
            None,
            (1e-2 ** 0.6, 0, 0, 0, 0),
            {
                "buffer_size": 5,
                "buffer_device": "cpu",
                "epsilon": 1e-2,
                "alpha": 0.6,
                "beta": 0.4,
                "beta_increment_per_sampling": 1e-3,
            },
        ),
    ]

    @pytest.mark.parametrize(
        "trans_list,priority,should_be_wt_leaf," "buffer_kwargs", param_test_append
    )
    def test_append(self, trans_list, priority, should_be_wt_leaf, buffer_kwargs):
        buffer = PrioritizedBuffer(**buffer_kwargs)
        for trans in trans_list:
            buffer.append(trans, priority=priority)
        assert np.all(buffer.wt_tree.get_leaf_all_weights() == should_be_wt_leaf)

    ########################################################################
    # Test for PrioritizedBuffer.clear
    ########################################################################
    param_test_clear = [
        (
            [
                {
                    "state": {"state_1": t.zeros([1, 2])},
                    "action": {"action_1": t.zeros([1, 3])},
                    "next_state": {"next_state_1": t.zeros([1, 2])},
                    "reward": 1,
                    "terminal": True,
                    "some_custom_attr": None,
                }
            ],
            1,
            {
                "buffer_size": 5,
                "buffer_device": "cpu",
                "epsilon": 1e-2,
                "alpha": 0.6,
                "beta": 0.4,
                "beta_increment_per_sampling": 1e-3,
            },
        ),
    ]

    @pytest.mark.parametrize("trans_list,priority,buffer_kwargs", param_test_clear)
    def test_clear(self, trans_list, priority, buffer_kwargs):
        buffer = PrioritizedBuffer(**buffer_kwargs)
        for trans in trans_list:
            buffer.append(trans, priority=priority)
        buffer.clear()

    ########################################################################
    # Test for PrioritizedBuffer.update_priority
    ########################################################################
    param_test_update_priority = [
        (
            [
                {
                    "state": {"state_1": t.zeros([1, 2])},
                    "action": {"action_1": t.zeros([1, 3])},
                    "next_state": {"next_state_1": t.zeros([1, 2])},
                    "reward": 1,
                    "terminal": True,
                    "some_custom_attr": None,
                }
            ],
            1,
            np.array([0]),
            np.array([2]),
            ((2 + 1e-2) ** 0.6, 0, 0, 0, 0),
            {
                "buffer_size": 5,
                "buffer_device": "cpu",
                "epsilon": 1e-2,
                "alpha": 0.6,
                "beta": 0.4,
                "beta_increment_per_sampling": 1e-3,
            },
        ),
    ]

    @pytest.mark.parametrize(
        "trans_list,priority,upd_indexes,"
        "upd_priorities,should_be_wt_leaf,"
        "buffer_kwargs",
        param_test_update_priority,
    )
    def test_update_priority(
        self,
        trans_list,
        priority,
        upd_indexes,
        upd_priorities,
        should_be_wt_leaf,
        buffer_kwargs,
    ):
        buffer = PrioritizedBuffer(**buffer_kwargs)
        for trans in trans_list:
            buffer.append(trans, priority=priority)
        buffer.update_priority(upd_priorities, upd_indexes)
        assert np.all(buffer.wt_tree.get_leaf_all_weights() == should_be_wt_leaf)

    ########################################################################
    # Test for PrioritizedBuffer.sample_batch
    ########################################################################
    empty_trans_list = []
    full_trans_list = [
        (
            {
                "state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.zeros([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1,
                "terminal": True,
                "index": 0,
            },
            1,
        ),
        (
            {
                "state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.zeros([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1,
                "terminal": True,
                "index": 1,
            },
            1,
        ),
        (
            {
                "state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.zeros([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1,
                "terminal": True,
                "index": 2,
            },
            1,
        ),
        (
            {
                "state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.zeros([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1,
                "terminal": True,
                "index": 3,
            },
            0.3,
        ),
        (
            {
                "state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.zeros([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1,
                "terminal": True,
                "index": 4,
            },
            0.3,
        ),
    ]
    param_test_sample_batch = [
        (  # test for empty batch & empty buffer
            empty_trans_list,
            0,
            None,
            None,
            None,
            {
                "buffer_size": 5,
                "buffer_device": "cpu",
                "epsilon": 1e-2,
                "alpha": 0.6,
                "beta": 0.4,
                "beta_increment_per_sampling": 1e-3,
            },
            {
                "batch_size": 5,
                "concatenate": True,
                "sample_attrs": ["index"],
                "additional_concat_attrs": ["index"],
            },
            0,
        ),
        (  # test for regular sample
            full_trans_list,
            5,
            [0, 1, 2, 2, 4],
            [0, 1, 2, 2, 4],
            [0.75316421, 0.75316421, 0.75316421, 0.75316421, 1.0],
            {
                "buffer_size": 5,
                "epsilon": 1e-2,
                "alpha": 0.6,
                "beta": 0.4,
                "beta_increment_per_sampling": 1e-3,
            },
            {
                "batch_size": 5,
                "concatenate": True,
                "sample_attrs": ["index"],
                "additional_concat_attrs": ["index"],
            },
            0,
        ),
        (  # test for device="cpu"
            full_trans_list,
            5,
            [0, 1, 2, 2, 4],
            [0, 1, 2, 2, 4],
            [0.75316421, 0.75316421, 0.75316421, 0.75316421, 1.0],
            {
                "buffer_size": 5,
                "epsilon": 1e-2,
                "alpha": 0.6,
                "beta": 0.4,
                "beta_increment_per_sampling": 1e-3,
            },
            {
                "batch_size": 5,
                "concatenate": True,
                "device": "cpu",
                "sample_attrs": ["index"],
                "additional_concat_attrs": ["index"],
            },
            0,
        ),
    ]

    @pytest.mark.parametrize(
        "trans_list,sampled_size,sampled_result,"
        "sampled_index,sampled_is_weight,buffer_kwargs,"
        "sample_kwargs,np_seed",
        param_test_sample_batch,
    )
    def test_sample_batch(
        self,
        trans_list,
        sampled_size,
        sampled_result,
        sampled_index,
        sampled_is_weight,
        buffer_kwargs,
        sample_kwargs,
        np_seed,
        pytestconfig,
    ):
        np.random.seed(np_seed)
        if "device" not in buffer_kwargs:
            buffer_kwargs["device"] = pytestconfig.getoption("gpu_device")
        buffer = PrioritizedBuffer(**buffer_kwargs)
        for trans, priority in trans_list:
            buffer.append(trans, priority=priority)

        bsize, result, index, is_weight = buffer.sample_batch(**sample_kwargs)
        assert bsize == sampled_size

        if sampled_result is not None:
            assert np.all(result[0].flatten().cpu().tolist() == sampled_result)
        else:
            assert result is None

        if sampled_index is not None:
            assert np.all(index == sampled_index)
        else:
            assert index is None

        if sampled_result is not None:
            assert np.all(np.abs(is_weight - sampled_is_weight) < 1e-6)
        else:
            assert is_weight is None
