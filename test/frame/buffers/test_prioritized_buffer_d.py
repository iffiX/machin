from machin.frame.buffers import DistributedPrioritizedBuffer
from test.util_run_multi import *

import random
import torch as t
import numpy as np


class TestDistributedPrioritizedBuffer(WorldTestBase):
    BUFFER_SIZE = 1
    SAMPLE_BUFFER_SIZE = 10

    ########################################################################
    # Test for DistributedPrioritizedBuffer.append and sample
    ########################################################################
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
                "reward": 2,
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
                "reward": 3,
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
                "reward": 4,
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
                "reward": 5,
                "terminal": True,
                "index": 4,
            },
            0.3,
        ),
    ]

    # test a normal sampling process, where p0 and p1 append to the buffer
    # periodically, and p2 sample from the buffer periodically.
    @staticmethod
    @run_multi(expected_results=[True, True, True], args_list=[(full_trans_list,)] * 3)
    @WorldTestBase.setup_world
    def test_append_sample_random(rank, trans_list):
        world = get_world()
        count = 0
        default_logger.info(f"{rank} started")
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            begin = time()
            while time() - begin < 10:
                trans, prior = random.choice(trans_list)
                buffer.append(trans)
                default_logger.info(f"{rank} append {count} success")
                count += 1
                sleep(random.random() * 0.5)
        else:
            sleep(5)
            begin = time()
            while time() - begin < 5:
                batch_size, sample, indexes, priorities = buffer.sample_batch(10)
                default_logger.info(f"sampled batch size: {batch_size}")
                assert batch_size > 0
                # state
                assert list(sample[0]["state_1"].shape) == [batch_size, 2]
                # action
                assert list(sample[1]["action_1"].shape) == [batch_size, 3]
                # next state
                assert list(sample[2]["next_state_1"].shape) == [batch_size, 2]
                # reward
                assert list(sample[3].shape) == [batch_size, 1]
                # terminal
                assert list(sample[4].shape) == [batch_size, 1]
                # index
                assert len(sample[5]) == batch_size
                # simulate perform a backward process
                sleep(1)
                buffer.update_priority(priorities, indexes)
                default_logger.info(f"{rank} sample {count} success")
                count += 1
                sleep(1)
        return True

    # controlled test sampling process, where p0 and p1 append to the buffer
    # periodically, and p2 sample from the buffer periodically. however, p0 and
    # p1 will finish appending before p2, so the test result is always the same.
    @staticmethod
    @run_multi(expected_results=[True, True, True], args_list=[(full_trans_list,)] * 3)
    @WorldTestBase.setup_world
    def test_append_sample_controlled(rank, trans_list):
        world = get_world()
        default_logger.info(f"{rank} started")
        np.random.seed(0)
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            for i in range(5):
                trans, prior = trans_list[i]
                buffer.append(trans, prior)
            sleep(5)
        else:
            sleep(2)
            batch_size, sample, indexes, priorities = buffer.sample_batch(
                10, sample_attrs=["index"]
            )
            default_logger.info(f"sampled batch size: {batch_size}")
            default_logger.info(sample)
            default_logger.info(indexes)
            default_logger.info(priorities)
            assert batch_size == 10
            assert sample[0] == [0, 1, 2, 2, 4, 0, 1, 2, 2, 4]
            assert list(indexes.keys()) == ["0", "1"]
            assert np.all(
                np.abs(
                    priorities
                    - [
                        0.75316421,
                        0.75316421,
                        0.75316421,
                        0.75316421,
                        1.0,
                        0.75316421,
                        0.75316421,
                        0.75316421,
                        0.75316421,
                        1.0,
                    ]
                )
                < 1e-6
            )
            buffer.update_priority(priorities, indexes)
        return True

    # sample from two empty buffers
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_sample_empty_buffer(rank):
        world = get_world()
        default_logger.info(f"{rank} started")
        np.random.seed(0)
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            sleep(5)
        else:
            sleep(2)
            batch_size, sample, indexes, priorities = buffer.sample_batch(
                10, sample_attrs=["index"]
            )
            assert batch_size == 0
            assert sample is None
            assert indexes is None
            assert priorities is None
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True], args_list=[(full_trans_list,)] * 3)
    @WorldTestBase.setup_world
    def test_append_sample_empty(rank, trans_list):
        world = get_world()
        default_logger.info(f"{rank} started")
        np.random.seed(0)
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            for i in range(5):
                trans, prior = trans_list[i]
                buffer.append(trans)
            sleep(5)
        else:
            sleep(2)
            batch_size, sample, indexes, priorities = buffer.sample_batch(0)
            assert batch_size == 0
            assert sample is None
            assert indexes is None
            assert priorities is None
        return True

    ########################################################################
    # Test for DistributedPrioritizedBuffer.size and all_size
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True], args_list=[(full_trans_list,)] * 3)
    @WorldTestBase.setup_world
    def test_append_size(rank, trans_list):
        world = get_world()
        default_logger.info(f"{rank} started")
        np.random.seed(0)
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            if rank == 0:
                for i in range(5):
                    data, prior = trans_list[i]
                    buffer.append(data, prior)
                assert buffer.size() == 5
            else:
                assert buffer.size() == 0
            sleep(5)
        else:
            sleep(2)
            assert buffer.size() == 0
            assert buffer.all_size() == 5
        return True

    ########################################################################
    # Test for DistributedPrioritizedBuffer.clear
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True], args_list=[(full_trans_list,)] * 3)
    @WorldTestBase.setup_world
    def test_append_clear(rank, trans_list):
        world = get_world()
        default_logger.info(f"{rank} started")
        np.random.seed(0)
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            for i in range(5):
                trans, prior = trans_list[i]
                buffer.append(trans, prior)
            if rank == 0:
                buffer.clear()
                assert buffer.size() == 0
            sleep(5)
        else:
            sleep(2)
            assert buffer.all_size() == 5
            buffer.all_clear()
            assert buffer.all_size() == 0
        return True
