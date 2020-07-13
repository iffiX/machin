from machin.parallel.distributed.role import RoleBase
from machin.parallel.distributed.world import (
    World,
    get_cur_role, get_cur_rank
)
from ..util_run_multi import *
import time
import torch as t


def watch(processes, result_watcher):
    while True:
        for p in processes[0]:
            p.watch()
        if result_watcher():
            break
        sleep(1e-3)


class WorkerService(object):
    _count = 0

    def counter(self):
        self._count += 1
        return self._count

    def get_count(self):
        return self._count


def worker_calculate(a, b):
    return a + b


class Worker(RoleBase):
    NAME = "Worker"

    def __init__(self, index):
        global world
        super(Worker, self).__init__(index)
        self.service = WorkerService()
        self.nn = t.nn.Linear(5, 5, bias=False)
        with t.no_grad():
            self.nn.weight.fill_(index)

        self.group = world.create_rpc_group("Company", roles=[
            ("Worker", 0), ("Worker", 1), ("Worker", 2),
            ("Manager", 0), ("Manager", 1), ("Manager", 2)
        ])

        # expose service to members in group
        self.group.rpc_register_paired("worker_service", self.service)
        self.group.rpc_register_paired("worker_nn", self.nn)

    def main(self):
        default_logger.info("Role {} started on Process[{}]"
                            .format(get_cur_role(), get_cur_rank()))
        time.sleep(20)
        self.group.destroy()


class WorkerLossy(Worker):
    NAME = "Worker"

    def __init__(self, index):
        if get_cur_rank() in (0, 2):
            default_logger.info("test crash Role {} on Process[{}]"
                                .format(get_cur_role(), get_cur_rank()))
            exit(0)
        super(WorkerLossy, self).__init__(index)


class Manager(RoleBase):
    NAME = "Manager"

    def __init__(self, index):
        global world
        super(Manager, self).__init__(index)
        self.service = WorkerService()
        group = world.create_rpc_group("Company", roles=[
            ("Worker", 0), ("Worker", 1), ("Worker", 2),
            ("Manager", 0), ("Manager", 1), ("Manager", 2)
        ])
        self.group = world.get_rpc_group("Company")

    def main(self):
        global success

        assert self.group.size() == 6
        assert self.group.is_member(("Worker", self.role_index))
        assert set(self.group.get_group_members()) == {
            ("Worker", 0), ("Worker", 1), ("Worker", 2),
            ("Manager", 0), ("Manager", 1), ("Manager", 2)
        }
        assert self.group.get_cur_role() == ("Manager", self.role_index)

        default_logger.info("Role {} started on Process[{}]"
                            .format(get_cur_role(), get_cur_rank()))

        # tell the respective worker to perform a simple calculation
        default_logger.info("Role {} begin test rpc normal sync"
                            .format(get_cur_role()))
        assert self.group.rpc_sync(
            to="Worker:{}".format(self.role_index),
            func=worker_calculate,
            args=(1, 2)
        ) == 3
        default_logger.info("Role {} begin test rpc normal async"
                            .format(get_cur_role()))
        assert self.group.rpc_async(
            to=("Worker", self.role_index),
            func=worker_calculate,
            args=(1, 2)
        ).wait() == 3
        default_logger.info("Role {} begin test rpc normal remote"
                            .format(get_cur_role()))
        assert self.group.rpc_remote(
            to=("Worker", self.role_index),
            func=worker_calculate,
            args=(1, 2)
        ).to_here() == 3

        # tell the respective worker to count
        default_logger.info("Role {} begin test rpc paired class sync"
                            .format(get_cur_role()))
        for i in range(10):
            self.group.rpc_paired_class_sync(
                to=("Worker", self.role_index),
                cls_method=WorkerService.counter,
                name="worker_service",
            )
        default_logger.info("Role {} begin test rpc paired class async"
                            .format(get_cur_role()))
        for i in range(10):
            self.group.rpc_paired_class_async(
                to=("Worker", self.role_index),
                cls_method=WorkerService.counter,
                name="worker_service",
            )
        default_logger.info("Role {} begin test rpc paired class remote"
                            .format(get_cur_role()))
        for i in range(10):
            self.group.rpc_paired_class_remote(
                to=("Worker", self.role_index),
                cls_method=WorkerService.counter,
                name="worker_service",
            )

        count = self.group.rpc_paired_class_remote(
            to=("Worker", self.role_index),
            cls_method=WorkerService.get_count,
            name="worker_service",
        ).to_here()
        assert count == 30

        # tell the respective worker to perform a neural network forward
        # operation
        default_logger.info("Role {} begin test rpc nn sync"
                            .format(get_cur_role()))
        assert t.sum(self.group.rpc_paired_nn_module_sync(
            to=("Worker", self.role_index),
            name="worker_nn",
            args=(t.ones([5]),)
        )) == 25 * self.role_index

        default_logger.info("Role {} begin test rpc nn async"
                            .format(get_cur_role()))
        assert t.sum(self.group.rpc_paired_nn_module_async(
            to=("Worker", self.role_index),
            name="worker_nn",
            args=(t.ones([5]),)
        ).wait()) == 25 * self.role_index

        default_logger.info("Role {} begin test rpc nn remote"
                            .format(get_cur_role()))
        assert t.sum(self.group.rpc_paired_nn_module_remote(
            to=("Worker", self.role_index),
            name="worker_nn",
            args=(t.ones([5]),)
        ).to_here()) == 25 * self.role_index

        default_logger.info("Role {} result check passed!"
                            .format(get_cur_role()))
        self.group.destroy()
        success[self.role_index] = True


class TestWorld(object):
    ########################################################################
    # Test routine for sub processes
    ########################################################################
    @classmethod
    def subproc_start_world(cls, rank, roles):
        # election function for all tests
        global world
        global success
        success = {}
        world = World(world_size=3, rank=rank, roles=roles,
                      rpc_timeout=0.5, election_timeout=0.3, logging=True)
        default_logger.info("World created on {}".format(rank))

    @classmethod
    def subproc_start_world_with_roles(cls, rank, run_time=10):
        global world, success
        begin = time.time()
        while time.time() - begin < run_time:
            world.watch()
            time.sleep(1e-1)
        return success

    ########################################################################
    # Test for collective communications
    ########################################################################
    @classmethod
    def subproc_test_world_coll_comm(cls, rank):
        global world
        world.watch()  # for coverage
        group = world.create_collective_group(ranks=[0, 1, 2])

        # test send and recv
        default_logger.info("test send and recv")
        if rank == 0:
            group.send(t.zeros([5]), 1)
            group.send(t.zeros([5]), 2)
        else:
            a = t.ones([5])
            assert group.recv(a) == 0
            assert t.all(a == 0)
        group.barrier()

        # test isend and irecv
        default_logger.info("test isend and irecv")
        if rank == 0:
            # need to wait otherwise won't execute
            group.isend(t.zeros([5]), 1).wait()
            group.isend(t.zeros([5]), 2).wait()
        else:
            a = t.ones([5])
            assert group.irecv(a).wait() == 0
            assert t.all(a == 0)
        group.barrier()

        # test broadcast
        default_logger.info("test broadcast")
        if rank == 0:
            a = t.ones([5])
        else:
            a = t.zeros([5])
        group.broadcast(a, 0)
        assert t.all(a == 1)
        group.barrier()

        # test all reduce
        default_logger.info("test all_reduce")
        a = t.full([5], rank)
        group.all_reduce(a)
        assert t.all(a == 3)
        group.barrier()

        # test reduce
        default_logger.info("test reduce")
        a = t.full([5], 5 - rank)
        group.reduce(a, 1)
        if rank == 1:
            assert t.all(a == 12)
        group.barrier()

        # test all gather
        default_logger.info("test all_gather")
        a = t.full([5], rank)
        a_list = [t.full([5], -1), t.full([5], -1), t.full([5], -1)]
        group.all_gather(a_list, a)
        assert t.all(a_list[0] == 0)
        assert t.all(a_list[1] == 1)
        assert t.all(a_list[2] == 2)
        group.barrier()

        # test gather
        default_logger.info("test gather")
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
        group.barrier()

        # test scatter
        default_logger.info("test scatter")
        if rank == 0:
            a_list = [t.full([5], 0), t.full([5], 1), t.full([5], 2)]
        else:
            a_list = None
        a = t.full([5], -1)
        group.scatter(a, a_list, 0)
        assert (t.all(a == 0) or
                t.all(a == 1) or
                t.all(a == 2))

        assert group.size() == 3
        group.destroy()

    def test_world_coll_comm(self, processes):
        result, watcher = run_multi(processes,
                                    self.subproc_start_world,
                                    args_list=[(dict(),)] * 3)
        watch(processes, watcher)
        default_logger.info("All world inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_world_coll_comm)
        watch(processes, watcher)

    ########################################################################
    # Test for rpc
    ########################################################################
    def test_rpc(self, processes):
        result, watcher = run_multi(processes,
                                    self.subproc_start_world,
                                    args_list=[(
                                        {"Worker": (Worker, 3),
                                         "Manager": (Manager, 3)},
                                    )] * 3)
        watch(processes, watcher)
        default_logger.info("All world inited")
        result, watcher = run_multi(processes,
                                    self.subproc_start_world_with_roles,
                                    args_list=[(5,)] * 3)
        watch(processes, watcher)
        success = {}
        for r in result:
            success.update(r)
        assert success == {0: True, 1: True, 2: True}

    def test_rpc_lossy(self, processes):
        result, watcher = run_multi(processes,
                                    self.subproc_start_world,
                                    args_list=[(
                                        {"Worker": (WorkerLossy, 3),
                                         "Manager": (Manager, 3)},
                                    )] * 3)
        watch(processes, watcher)
        default_logger.info("All world inited")
        result, watcher = run_multi(processes,
                                    self.subproc_start_world_with_roles,
                                    args_list=[(30,)] * 3)
        watch(processes, watcher)
        success = {}
        for r in result:
            success.update(r)
        assert success == {0: True, 1: True, 2: True}
