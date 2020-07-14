from test.util_run_multi import *
import torch.nn as nn
import torch as t


class WorkerModel(nn.Module):
    def __init__(self):
        super(WorkerModel, self).__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)
        with t.no_grad():
            self.fc1.weight.fill_(1)

    def forward(self, x):
        return self.fc1(x)


class WorkerService(object):
    _count = 0

    def counter(self):
        self._count += 1
        return self._count

    def get_count(self):
        return self._count


def worker_calculate(a, b):
    return a + b


class TestWorld(WorldTestBase):
    ########################################################################
    # Test for collective communications
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_cc_send_recv(rank):
        world = get_world()
        group = world.create_collective_group(ranks=[0, 1, 2])

        if rank == 0:
            group.send(t.zeros([5]), 1)
            group.send(t.zeros([5]), 2)
        else:
            a = t.ones([5])
            assert group.recv(a) == 0
            assert t.all(a == 0)
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_cc_isend_irecv(rank):
        world = get_world()
        group = world.create_collective_group(ranks=[0, 1, 2])
        if rank == 0:
            # need to wait otherwise won't execute
            group.isend(t.zeros([5]), 1).wait()
            group.isend(t.zeros([5]), 2).wait()
        else:
            a = t.ones([5])
            assert group.irecv(a).wait() == 0
            assert t.all(a == 0)
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_cc_broadcast(rank):
        world = get_world()
        group = world.create_collective_group(ranks=[0, 1, 2])
        # test broadcast
        default_logger.info("test broadcast")
        if rank == 0:
            a = t.ones([5])
        else:
            a = t.zeros([5])
        group.broadcast(a, 0)
        assert t.all(a == 1)
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_cc_all_reduce(rank):
        world = get_world()
        group = world.create_collective_group(ranks=[0, 1, 2])
        a = t.full([5], rank)
        group.all_reduce(a)
        assert t.all(a == 3)
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_cc_reduce(rank):
        world = get_world()
        group = world.create_collective_group(ranks=[0, 1, 2])
        a = t.full([5], 5 - rank)
        group.reduce(a, 1)
        if rank == 1:
            assert t.all(a == 12)
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_cc_all_gather(rank):
        world = get_world()
        group = world.create_collective_group(ranks=[0, 1, 2])
        a = t.full([5], rank)
        a_list = [t.full([5], -1), t.full([5], -1), t.full([5], -1)]
        group.all_gather(a_list, a)
        assert t.all(a_list[0] == 0)
        assert t.all(a_list[1] == 1)
        assert t.all(a_list[2] == 2)
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_cc_gather(rank):
        world = get_world()
        group = world.create_collective_group(ranks=[0, 1, 2])
        a = t.full([5], rank)
        if rank == 1:
            a_list = [t.full([5], -1), t.full([5], -1), t.full([5], -1)]
        else:
            a_list = None
        group.gather(a, a_list, 1)
        if rank == 1:
            assert t.all(a_list[0] == 0)
            assert t.all(a_list[1] == 1)
            assert t.all(a_list[2] == 2)
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_cc_scatter(rank):
        world = get_world()
        group = world.create_collective_group(ranks=[0, 1, 2])
        if rank == 0:
            a_list = [t.full([5], 0), t.full([5], 1), t.full([5], 2)]
        else:
            a_list = None
        a = t.full([5], -1)
        group.scatter(a, a_list, 0)
        assert (t.all(a == 0) or
                t.all(a == 1) or
                t.all(a == 2))
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_cc_barrier(_):
        world = get_world()
        group = world.create_collective_group(ranks=[0, 1, 2])
        assert group.size() == 3
        group.barrier()
        group.destroy()
        return True

    ########################################################################
    # Test for rpc
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_sync(_):
        world = get_world()
        group = world.create_rpc_group("group", ["0", "1", "2"])
        assert (group.rpc_sync("1", worker_calculate, args=(1, 2), timeout=1)
                == 3)
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_async(_):
        world = get_world()
        group = world.create_rpc_group("group", ["0", "1", "2"])
        assert (group.rpc_async("1", worker_calculate, args=(1, 2), timeout=1)
                .wait() == 3)
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_remote(_):
        world = get_world()
        group = world.create_rpc_group("group", ["0", "1", "2"])
        assert (group.remote("1", worker_calculate, args=(1, 2), timeout=1)
                .to_here() == 3)
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_paired_cls_sync(rank):
        world = get_world()
        if rank == 0:
            group = world.create_rpc_group("group", ["0", "1"])
            group.rpc_pair("service", WorkerService())
            sleep(5)
            group.destroy()
        elif rank == 1:
            group = world.create_rpc_group("group", ["0", "1"])
            for i in range(10):
                group.rpc_paired_class_sync("0", "service",
                                            WorkerService.counter,
                                            timeout=1)
            group.destroy()
        elif rank == 2:
            group = world.get_rpc_group("group", "0")
            sleep(2)
            assert group.rpc_paired_class_sync("0", "service",
                                               WorkerService.get_count,
                                               timeout=1) == 10
            group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_paired_cls_async(rank):
        world = get_world()
        if rank == 0:
            group = world.create_rpc_group("group", ["0", "1"])
            group.rpc_pair("service", WorkerService())
            sleep(5)
            group.destroy()
        elif rank == 1:
            group = world.create_rpc_group("group", ["0", "1"])
            for i in range(10):
                group.rpc_paired_class_async("0", "service",
                                             WorkerService.counter,
                                             timeout=1).wait()
            group.destroy()
        elif rank == 2:
            group = world.get_rpc_group("group", "0")
            sleep(2)
            assert group.rpc_paired_class_async("0", "service",
                                                WorkerService.get_count,
                                                timeout=1).wait() == 10
            group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_paired_cls_remote(rank):
        world = get_world()
        if rank == 0:
            group = world.create_rpc_group("group", ["0", "1"])
            group.rpc_pair("service", WorkerService())
            sleep(5)
            group.destroy()
        elif rank == 1:
            group = world.create_rpc_group("group", ["0", "1"])
            for i in range(10):
                group.rpc_paired_class_remote("0", "service",
                                              WorkerService.counter,
                                              timeout=1).to_here()
            group.destroy()
        elif rank == 2:
            group = world.get_rpc_group("group", "0")
            sleep(2)
            assert group.rpc_paired_class_remote("0", "service",
                                                 WorkerService.get_count,
                                                 timeout=1).to_here() == 10
            group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_paired_nn_sync(rank):
        world = get_world()
        if rank == 0:
            group = world.create_rpc_group("group", ["0"])
            group.rpc_pair("model", WorkerModel())
            sleep(3)
            group.destroy()
        else:
            group = world.get_rpc_group("group", "0")
            assert (group.rpc_paired_model_sync(
                "0", "model", args=(t.ones([1, 1])), timeout=1).item()
                    == 1)
            group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_paired_nn_async(rank):
        world = get_world()
        if rank == 0:
            group = world.create_rpc_group("group", ["0"])
            group.rpc_pair("model", WorkerModel())
            sleep(3)
            group.destroy()
        else:
            group = world.get_rpc_group("group", "0")
            assert (group.rpc_paired_model_async(
                "0", "model", args=(t.ones([1, 1])), timeout=1).wait().item()
                    == 1)
            group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_paired_nn_remote(rank):
        world = get_world()
        if rank == 0:
            group = world.create_rpc_group("group", ["0"])
            group.rpc_pair("model", WorkerModel())
            sleep(3)
            group.destroy()
        else:
            group = world.get_rpc_group("group", "0")
            assert (group.rpc_paired_model_remote(
                "0", "model", args=(t.ones([1, 1])), timeout=1).to_here().item()
                    == 1)
            group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_misc(_):
        world = get_world()
        group = world.create_rpc_group("group", ["0", "1", "2"])
        assert group.size() == 3
        assert group.is_member("0")
        assert group.get_group_members() == ["0", "1", "2"]
        return True
