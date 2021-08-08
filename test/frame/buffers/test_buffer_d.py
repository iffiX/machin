from machin.frame.buffers import DistributedBuffer
from test.util_run_multi import *
from test.util_platforms import linux_only_forall

import torch as t

linux_only_forall()


class TestDistributedBuffer:
    BUFFER_SIZE = 1
    SAMPLE_BUFFER_SIZE = 10

    ########################################################################
    # Test for DistributedBuffer.store_episode and sample_batch
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @setup_world
    def test_store_episode_and_sample_batch(rank):
        world = get_world()
        episode = [
            {
                "state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.ones([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1.5,
                "terminal": True,
            }
        ] * 5
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedBuffer("buffer", group, 5)
        if rank in (0, 1):
            buffer.store_episode(episode)
        else:
            sleep(2)
            batch_size, sample = buffer.sample_batch(7)
            assert batch_size > 0
            assert len(sample) == 5
            # state
            assert t.all(sample[0]["state_1"] == 0) and list(
                sample[0]["state_1"].shape
            ) == [batch_size, 2]
            # reward
            assert t.all(sample[3] == 1.5) and list(sample[3].shape) == [batch_size, 1]
            # terminal
            assert t.all(sample[4]) and list(sample[4].shape) == [batch_size, 1]
        return True

    ########################################################################
    # Test for DistributedBuffer.size and all_size
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @setup_world
    def test_size_and_all_size(rank):
        world = get_world()
        episode = [
            {
                "state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.ones([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1.5,
                "terminal": True,
            }
        ] * 5
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedBuffer("buffer", group, 5)
        if rank in (0, 1):
            if rank == 0:
                buffer.store_episode(episode)
                assert buffer.size() == 5
            sleep(5)
        else:
            sleep(2)
            assert buffer.size() == 0
            assert buffer.all_size() == 5
        return True

    ########################################################################
    # Test for DistributedBuffer.clear
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @setup_world
    def test_clear(rank):
        world = get_world()
        episode = [
            {
                "state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.ones([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1.5,
                "terminal": True,
            }
        ] * 5
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedBuffer("buffer", group, 5)
        if rank in (0, 1):
            buffer.store_episode(episode)
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
