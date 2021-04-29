from machin.parallel.distributed import get_cur_name, get_cur_rank
from test.util_run_multi import *
from test.util_fixtures import *
from test.util_platforms import linux_only_forall
import torch as t

linux_only_forall()


class WorkerService:
    counter = 0

    def count(self):
        self.counter += 1
        return self.counter

    def get_count(self):
        return self.counter


def worker_calculate(a, b):
    return a + b


class TestWorld(WorldTestBase):
    ########################################################################
    # Test for world APIs
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_get_info(rank):
        _world = get_world()
        assert get_cur_rank() == rank
        assert get_cur_name() == str(rank)
        return True

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
        assert t.all(a == 0) or t.all(a == 1) or t.all(a == 2)
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

    @staticmethod
    @run_multi(expected_results=[True, True, True], pass_through=["gpu"])
    @WorldTestBase.setup_world
    def test_cc_broadcast_multigpu(rank, gpu):
        if isinstance(gpu, str) and gpu.startswith("cuda"):
            world = get_world()
            group = world.create_collective_group(ranks=[0, 1, 2])
            if rank == 0:
                a = [t.ones([5], device=gpu)]
            else:
                a = [t.zeros([5], device=gpu)]
            group.broadcast_multigpu(a, 0)
            assert t.all(a[0] == 1)
            group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True], pass_through=["gpu"])
    @WorldTestBase.setup_world
    def test_cc_all_reduce_multigpu(_, gpu):
        if isinstance(gpu, str) and gpu.startswith("cuda"):
            world = get_world()
            group = world.create_collective_group(ranks=[0, 1, 2])
            a = [t.ones([5], device=gpu)]
            group.all_reduce_multigpu(a)
            assert t.all(a[0] == 3)
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
        assert group.rpc_sync("1", worker_calculate, args=(1, 2), timeout=1) == 3
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_async(_):
        world = get_world()
        group = world.create_rpc_group("group", ["0", "1", "2"])
        assert (
            group.rpc_async("1", worker_calculate, args=(1, 2), timeout=1).wait() == 3
        )
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_remote(_):
        world = get_world()
        group = world.create_rpc_group("group", ["0", "1", "2"])
        assert (
            group.remote("1", worker_calculate, args=(1, 2), timeout=1).to_here() == 3
        )
        group.destroy()
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_pair(rank):
        world = get_world()
        service = WorkerService()
        if rank == 0:
            group = world.create_rpc_group("group", ["0", "1"])
            service.counter = 20
            group.pair("service", service)

            # cannot pair an already used key
            with pytest.raises(KeyError, match="already paired to Group"):
                group.pair("service", service)

            sleep(4)
            group.unpair("service")
            group.destroy()
        elif rank == 1:
            group = world.create_rpc_group("group", ["0", "1"])
            sleep(2)
            assert group.is_paired("service")

            # cannot pair an already used key
            with pytest.raises(KeyError, match="already paired to Group"):
                group.pair("service", service)

            # cannot unpair the key paired by another process
            with pytest.raises(KeyError, match="not paired to"):
                group.unpair("service")

            # cannot find a not paired value
            with pytest.raises(KeyError, match="not found on Group"):
                group.get_paired("service2")

            assert group.get_paired("service").to_here().counter == 20
            sleep(4)
            assert not group.is_paired("service")
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_register(rank):
        world = get_world()
        service = WorkerService()
        if rank == 0:
            group = world.create_rpc_group("group", ["0", "1"])
            service.counter = 20
            group.register("count", service.count)
            group.register("get_count", service.get_count)

            # cannot register an already used key
            with pytest.raises(KeyError, match="already registered in Group"):
                group.register("count", service.count)

            sleep(4)
            assert service.get_count() == 23
            group.deregister("count")
            group.deregister("get_count")
            group.destroy()
        elif rank == 1:
            group = world.create_rpc_group("group", ["0", "1"])
            sleep(2)
            assert group.is_registered("count") and group.is_registered("get_count")

            # cannot register an already used key
            with pytest.raises(KeyError, match="already registered in Group"):
                group.register("count", service.count)

            # cannot deregister the key registered by another process
            with pytest.raises(KeyError, match="not registered in"):
                group.deregister("count")

            # cannot find a not registered service
            with pytest.raises(KeyError, match="not found on Group"):
                group.registered_sync("service2", args=())

            assert group.registered_sync("count") == 21
            assert group.registered_async("count").wait() == 22
            assert group.registered_remote("count").to_here() == 23
            sleep(4)
            assert not group.is_registered("count") and not group.is_registered(
                "get_count"
            )
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_get_info(rank):
        world = get_world()
        group = world.create_rpc_group("group", ["0", "1", "2"])
        assert group.size() == 3
        assert group.is_member("0")
        assert group.is_member()
        assert group.get_group_members() == ["0", "1", "2"]
        assert group.get_cur_name() == str(rank)
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_rpc_barrier(_):
        world = get_world()
        group = world.create_rpc_group("group", ["0", "1", "2"])
        assert group.size() == 3
        group.barrier()
        group.destroy()
        return True
