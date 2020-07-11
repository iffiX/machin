from machin.parallel.distributed import (
    RoleDispatcherSimple,
    RoleDispatcherElection,
    ElectionGroupStableRpc
)
from ..util_run_multi import *
import itertools
import pytest
import time


@pytest.fixture(scope="function")
def rpc_mocker():
    rpc_mocker = RpcNoLog(process_num=3, rpc_timeout=60,
                          rpc_init_wait_time=0.5,
                          print_expired=False,
                          rpc_response_time=[1e-3, 1e-2])
    rpc_mocker.start()
    yield rpc_mocker
    rpc_mocker.stop()
    default_logger.info("rpc_mocker stopped")


class RpcNoLog(RpcMocker):
    drop_match = lambda *_: False

    @staticmethod
    def interposer(obj, timestamp, src, to, fuzz, token):
        cmd, obj, timestamp, src, to, fuzz, token = \
            RpcMocker.interposer(obj, timestamp, src, to, fuzz, token)
        if Rpc.drop_match(obj, src, to):
            cmd = "drop"
        return cmd, obj, timestamp, src, to, fuzz, token

    @classmethod
    def set_drop_match(cls, drop_match=None):
        if drop_match is None:
            cls.drop_match = lambda *_: False
        else:
            cls.drop_match = drop_match


class TestRoleDispatcherSimple(object):
    def test_simple(self):
        dispatcher = RoleDispatcherSimple(rank=4, world_size=5,
                                          roles=["worker", "hauler", "soldier"],
                                          role_counts=[1, 2, 1])
        dispatcher.start()
        assert dispatcher.get_roles() == []
        assert dispatcher.get_rank(("worker", 0)) == 0
        assert dispatcher.get_rank(("hauler", 1)) == 2
        assert dispatcher.get_rank(("soldier", 0)) == 3
        assert dispatcher.get_rank(("invalid_role", 0)) is None
        assert dispatcher.get_rank(("worker", 10)) is None

        dispatcher = RoleDispatcherSimple(rank=3, world_size=4,
                                          roles=["worker", "hauler", "soldier"],
                                          role_counts=[1, 2, 1])
        assert dispatcher.get_roles() == [("soldier", 0)]


class TestRoleDispatcherElection(RpcTestBase):
    TIMEOUT_DELTA = 5e-1

    ########################################################################
    # Test routine for sub processes
    ########################################################################
    @classmethod
    def subproc_test_dispatch(cls, rank, run_time=10):
        elect_group = ElectionGroupStableRpc(name="elect_group",
                                             member_ranks=[0, 1, 2],
                                             rank=rank,
                                             timeout=cls.TIMEOUT_DELTA,
                                             logging=True)
        dispatcher = RoleDispatcherElection(name="dispatcher",
                                            rank=rank,
                                            world_size=3,
                                            roles=["worker", "hauler",
                                                   "soldier"],
                                            role_counts=[1, 2, 2],
                                            election_group=elect_group,
                                            logging=True)
        dispatcher.start()
        begin = time.time()
        while time.time() - begin < cls.TIMEOUT_DELTA * run_time:
            time.sleep(cls.TIMEOUT_DELTA / 1e3)
            dispatcher.watch()
        roles = dispatcher.get_roles()
        ranks = []
        for r in [("worker", 0), ("hauler", 0), ("hauler", 1),
                  ("soldier", 0), ("soldier", 1)]:
            ranks.append(dispatcher.get_rank(r))
        invalid_rank = dispatcher.get_rank(("soldier", 10))
        dispatcher.stop()
        return roles, ranks, invalid_rank

    ########################################################################
    # Test for RoleDispatcherElection under normal network condition
    ########################################################################
    def test_dispatch_normal(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker))
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_dispatch)
        watch(rpc_mocker, processes, watcher)
        roles_map = {
            rank: r[0] for rank, r in zip([0, 1, 2], result)
        }
        assert (set(itertools.chain(*roles_map.values())) ==
                {("worker", 0), ("hauler", 0), ("hauler", 1),
                 ("soldier", 0), ("soldier", 1)})

        # check route consistency on all processes
        assert (result[0][1] == result[1][1] and
                result[0][1] == result[2][1])
        assert (result[0][2] is None and
                result[1][2] is None and
                result[2][2] is None)
        default_logger.info("Assigned roles: {}".format(roles_map))

    ########################################################################
    # Test for RoleDispatcherElection where leader 0 is dropped
    ########################################################################
    def test_dispatch_drop_leader(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker))
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_dispatch,
                                    args_list=[(20,)] * 3)

        begin = time.time()
        while True:
            for p in processes[0]:
                p.watch()
            if time.time() - begin >= self.TIMEOUT_DELTA * 9:
                Rpc.drop_match = lambda _, src, to: to == 0 or src == 0
            if watcher():
                break

        roles_map = {
            rank: r[0] for rank, r in zip([0, 1, 2], result)
        }
        assert (set(itertools.chain(roles_map[1], roles_map[2])) ==
                {("worker", 0), ("hauler", 0), ("hauler", 1),
                 ("soldier", 0), ("soldier", 1)})

        # check route consistency on process 1 and 2 (0 is disconnected)

        assert result[1][1] == result[2][1]
        default_logger.info("Assigned roles: {}".format(roles_map))