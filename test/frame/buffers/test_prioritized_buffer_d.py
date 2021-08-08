from machin.frame.buffers import DistributedPrioritizedBuffer
from test.util_run_multi import *
from test.util_platforms import linux_only_forall

import random
import torch as t
import numpy as np


linux_only_forall()


class TestDistributedPrioritizedBuffer:
    BUFFER_SIZE = 1
    SAMPLE_BUFFER_SIZE = 10

    ########################################################################
    # Test for DistributedPrioritizedBuffer.store_episode and sample_batch
    ########################################################################
    full_episode = [
        {
            "state": {"state_1": t.zeros([1, 2])},
            "action": {"action_1": t.zeros([1, 3])},
            "next_state": {"next_state_1": t.zeros([1, 2])},
            "reward": 1,
            "terminal": True,
            "index": 0,
        },
        {
            "state": {"state_1": t.zeros([1, 2])},
            "action": {"action_1": t.zeros([1, 3])},
            "next_state": {"next_state_1": t.zeros([1, 2])},
            "reward": 1,
            "terminal": True,
            "index": 1,
        },
        {
            "state": {"state_1": t.zeros([1, 2])},
            "action": {"action_1": t.zeros([1, 3])},
            "next_state": {"next_state_1": t.zeros([1, 2])},
            "reward": 1,
            "terminal": True,
            "index": 2,
        },
        {
            "state": {"state_1": t.zeros([1, 2])},
            "action": {"action_1": t.zeros([1, 3])},
            "next_state": {"next_state_1": t.zeros([1, 2])},
            "reward": 1,
            "terminal": True,
            "index": 3,
        },
        {
            "state": {"state_1": t.zeros([1, 2])},
            "action": {"action_1": t.zeros([1, 3])},
            "next_state": {"next_state_1": t.zeros([1, 2])},
            "reward": 1,
            "terminal": True,
            "index": 4,
        },
    ]
    full_priorities = [1, 1, 1, 0.3, 0.3]

    # test a normal sampling process, where p0 and p1 store to the buffer
    # periodically, and p2 sample from the buffer periodically.
    @staticmethod
    @run_multi(
        expected_results=[True, True, True],
        args_list=[(full_episode, full_priorities)] * 3,
    )
    @setup_world
    def test_store_episode_and_sample_batch_random(rank, episode, priorities):
        world = get_world()
        count = 0
        default_logger.info(f"{rank} started")
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            begin = time()
            while time() - begin < 10:
                buffer.store_episode(episode, priorities=priorities)
                default_logger.info(f"{rank} store episode {count} success")
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
                # simulate the time to perform a backward process
                sleep(1)
                buffer.update_priority(priorities, indexes)
                default_logger.info(f"{rank} sample {count} success")
                count += 1
                sleep(1)
        return True

    # controlled test sampling process, where p0 and p1 store to the buffer
    # periodically, and p2 sample from the buffer periodically. however, p0 and
    # p1 will finish storing before p2, so the test result is always the same.
    @staticmethod
    @run_multi(
        expected_results=[True, True, True],
        args_list=[(full_episode, full_priorities)] * 3,
    )
    @setup_world
    def test_store_episode_and_sample_batch_controlled(
        rank, episode, priorities,
    ):
        world = get_world()
        default_logger.info(f"{rank} started")
        np.random.seed(0)
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            buffer.store_episode(episode, priorities=priorities)
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
    @setup_world
    def test_store_episode_and_sample_batch_from_empty(rank):
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
    @run_multi(
        expected_results=[True, True, True],
        args_list=[(full_episode, full_priorities)] * 3,
    )
    @setup_world
    def test_store_episode_and_sample_empty_batch(rank, episode, priorities):
        world = get_world()
        default_logger.info(f"{rank} started")
        np.random.seed(0)
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            buffer.store_episode(episode, priorities=priorities)
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
    @run_multi(
        expected_results=[True, True, True],
        args_list=[(full_episode, full_priorities)] * 3,
    )
    @setup_world
    def test_size_and_all_size(rank, episode, priorities):
        world = get_world()
        default_logger.info(f"{rank} started")
        np.random.seed(0)
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            if rank == 0:
                buffer.store_episode(episode, priorities=priorities)
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
    @run_multi(
        expected_results=[True, True, True],
        args_list=[(full_episode, full_priorities)] * 3,
    )
    @setup_world
    def test_clear(rank, episode, priorities):
        world = get_world()
        default_logger.info(f"{rank} started")
        np.random.seed(0)
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedPrioritizedBuffer("buffer", group, 5)
        if rank in (0, 1):
            buffer.store_episode(episode, priorities=priorities)
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
