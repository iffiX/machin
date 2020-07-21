from machin.frame.buffers import DistributedBuffer
from test.util_run_multi import *

import torch as t


class TestDistributedBuffer(WorldTestBase):
    BUFFER_SIZE = 1
    SAMPLE_BUFFER_SIZE = 10

    ########################################################################
    # Test for DistributedBuffer.append and sample
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_append_sample(rank):
        world = get_world()
        data = {"state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.ones([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1.5,
                "terminal": True}
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedBuffer("buffer", group, 5)
        if rank in (0, 1):
            begin = time()
            while time() - begin < 5:
                buffer.append(data)
                sleep(0.01)
        else:
            sleep(2)
            batch_size, sample = buffer.sample_batch(7)
            assert batch_size > 0
            assert len(sample) == 5
            # state
            assert (t.all(sample[0]["state_1"] == 0) and
                    list(sample[0]["state_1"].shape) == [batch_size, 2])
            # reward
            assert (t.all(sample[3] == 1.5) and
                    list(sample[3].shape) == [batch_size, 1])
            # terminal
            assert (t.all(sample[4]) and
                    list(sample[4].shape) == [batch_size, 1])
        return True

    ########################################################################
    # Test for DistributedBuffer.size and all_size
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_append_size(rank):
        world = get_world()
        data = {"state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.ones([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1.5,
                "terminal": True}
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedBuffer("buffer", group, 5)
        if rank in (0, 1):
            if rank == 0:
                for _ in range(5):
                    buffer.append(data)
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
    @WorldTestBase.setup_world
    def test_append_clear(rank):
        world = get_world()
        data = {"state": {"state_1": t.zeros([1, 2])},
                "action": {"action_1": t.ones([1, 3])},
                "next_state": {"next_state_1": t.zeros([1, 2])},
                "reward": 1.5,
                "terminal": True}
        group = world.create_rpc_group("group", ["0", "1", "2"])
        buffer = DistributedBuffer("buffer", group, 5)
        if rank in (0, 1):
            for i in range(5):
                buffer.append(data)
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
