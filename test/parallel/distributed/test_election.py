from machin.parallel.distributed import (
    ElectionGroupSimple,
    ElectionGroupStableRpc
)
from machin.parallel.process import Process
from machin.utils.logging import default_logger, INFO
from ..util_rpc_mocker import RpcMocker
from torch.distributed import rpc
from typing import Any, List
import dill
import mock
import time
import pytest
import random
import itertools
import multiprocessing as mp

# process number is 3
elect_group = None


def process_main(pipe):
    while True:
        func, args, kwargs = pipe.recv()
        pipe.send(func(*args, **kwargs))


@pytest.fixture(scope="function")
def processes():
    ctx = mp.get_context("fork")
    pipes = [mp.Pipe(duplex=True) for _ in [0, 1, 2]]
    processes = [Process(target=process_main, args=(pipes[i][0],), ctx=ctx)
                 for i in [0, 1, 2]]
    for p in processes:
        p.daemon = True
        p.start()
    yield processes, [pi[1] for pi in pipes]
    for p, pi, i in zip(processes, pipes, [0, 1, 2]):
        # try graceful shutdown first
        pi[1].send((exit, 0, {}))
        p.join(timeout=1e-1)
        if p.is_alive():
            # ungraceful shutdown
            default_logger.info("processes {} ungraceful shutdown".format(i))
            p.terminate()
            p.join()
    default_logger.info("processes stopped")


@pytest.fixture(scope="function")
def rpc_mocker():
    rpc_mocker = Rpc(process_num=3, rpc_timeout=60, rpc_init_wait_time=0.5,
                     print_expired=False, rpc_response_time=[1e-3, 5e-2])
    rpc_mocker.start()
    yield rpc_mocker
    rpc_mocker.stop()
    default_logger.info("rpc_mocker stopped")


def run_multi(processes, func, args_list=None, kwargs_list=None):
    procs, proc_pipes = processes
    args_list = (args_list
                 if args_list is not None
                 else itertools.repeat([]))
    kwargs_list = (kwargs_list
                   if kwargs_list is not None
                   else itertools.repeat({}))
    for pi, rank, args, kwargs in zip(proc_pipes, [0, 1, 2],
                                      args_list, kwargs_list):
        pi.send((func, [rank] + list(args), kwargs))

    result = [None, None, None]  # type: List[Any, Any, Any]
    finished = [False, False, False]

    def result_watcher():
        for p, pi, i in zip(procs, proc_pipes, [0, 1, 2]):
            if pi.poll(timeout=1e-3):
                result[i] = pi.recv()
                finished[i] = True
        return all(finished)

    return result, result_watcher


def watch(rpc_mocker, processes, result_watcher):
    while True:
        rpc_mocker.watch()
        for p in processes[0]:
            p.watch()
        if result_watcher():
            break


class Rpc(RpcMocker):
    _lock = mp.Lock()
    drop_match = lambda *_: False

    @staticmethod
    def interposer(obj, timestamp, src, to, fuzz, token):
        cmd, obj, timestamp, src, to, fuzz, token = \
            RpcMocker.interposer(obj, timestamp, src, to, fuzz, token)
        if Rpc.drop_match(obj, src, to):
            cmd = "drop"
        args = obj[2]
        with Rpc._lock:
            if cmd == "ok":
                default_logger.info(
                    "Rpc request from Process {} to {}, "
                    "rank={}, name={}, message={}".format(
                        src, to, args[1], args[2], args[3]
                    ))
            elif cmd == "drop":
                default_logger.info(
                    "Rpc request dropped from Process {} to {}, "
                    "rank={}, name={}, message={}".format(
                        src, to, args[1], args[2], args[3]
                    ))
        return cmd, obj, timestamp, src, to, fuzz, token

    @classmethod
    def set_drop_match(cls, drop_match=None):
        if drop_match is None:
            cls.drop_match = lambda *_: False
        else:
            cls.drop_match = drop_match

class TestElectionGroupSimple(object):
    ########################################################################
    # Test for ElectionGroupSimple.get_leader
    ########################################################################
    @staticmethod
    def subproc_test_get_leader(rank):
        global elect_group
        elect_group = ElectionGroupSimple(member_ranks=[0, 1, 2],
                                          rank=rank,
                                          leader_rank=0)
        elect_group.start()
        # no need to delay, immediate completion
        elect_group.watch()
        leader = elect_group.get_leader()
        elect_group.stop()
        return leader

    def test_get_leader(self, processes):
        result, watcher = run_multi(processes, self.subproc_test_get_leader)
        while not watcher():
            continue
        assert result == [0, 0, 0]

    ########################################################################
    # Test for AElectionGroupSimple.is_leader
    ########################################################################
    @staticmethod
    def subproc_test_is_leader(rank):
        global elect_group
        elect_group = ElectionGroupSimple(member_ranks=[0, 1, 2],
                                          rank=rank,
                                          leader_rank=0)
        elect_group.start()
        # no need to delay, immediate completion
        elect_group.watch()
        is_leader = elect_group.is_leader()
        elect_group.stop()
        return is_leader

    def test_is_leader(self, processes):
        result, watcher = run_multi(processes, self.subproc_test_is_leader)
        while not watcher():
            continue
        assert result == [True, False, False]

    ########################################################################
    # Test for ElectionGroupSimple.get_members
    ########################################################################
    @staticmethod
    def subproc_test_get_members(rank):
        global elect_group
        elect_group = ElectionGroupSimple(member_ranks=[0, 1, 2],
                                          rank=rank,
                                          leader_rank=0)
        elect_group.start()
        # no need to delay, immediate completion
        elect_group.watch()
        members = elect_group.get_members()
        elect_group.stop()
        return members

    def test_get_members(self, processes):
        result, watcher = run_multi(processes, self.subproc_test_get_members)
        while not watcher():
            continue
        assert result == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]


class TestElectionGroupStableRpc(object):
    # Stable Election Algorithm is first tested under the normal condition
    # with no lossy / slow rpc links,
    # and then tested with the fuzzy testing method
    TIMEOUT_DELTA = 1e-1

    ########################################################################
    # Test routine for sub processes
    ########################################################################
    @classmethod
    def subproc_test_elect(cls, rank, run_time=9):
        # election function for all tests
        global elect_group
        elect_group = ElectionGroupStableRpc(name="elect_group",
                                             member_ranks=[0, 1, 2],
                                             rank=rank,
                                             leader_rank=0,
                                             timeout=cls.TIMEOUT_DELTA)
        elect_group.logger.setLevel(INFO)
        default_logger.info("Start election group on {}".format(rank))
        elect_group.start()
        time.sleep(cls.TIMEOUT_DELTA * run_time)
        elect_group.watch()
        is_leader = elect_group.is_leader()
        leader = elect_group.get_leader()
        members = elect_group.get_members()
        elect_group.stop()
        return is_leader, leader, members

    ########################################################################
    # Test routine for sub processes, can collect multiple samples
    ########################################################################
    @classmethod
    def subproc_test_elect_multi(cls, rank, run_time=9, log_times=()):
        # election function for all tests
        global elect_group
        elect_group = ElectionGroupStableRpc(name="elect_group",
                                             member_ranks=[0, 1, 2],
                                             rank=rank,
                                             leader_rank=0,
                                             timeout=cls.TIMEOUT_DELTA)
        elect_group.logger.setLevel(INFO)
        default_logger.info("Start election group on {}".format(rank))

        begin = time.time()
        elect_group.start()

        log = []
        log_times = list(log_times)
        while time.time() - begin < cls.TIMEOUT_DELTA * run_time:
            time.sleep(cls.TIMEOUT_DELTA / 1000)
            elect_group.watch()
            if (log_times and
                    time.time() - begin >= cls.TIMEOUT_DELTA * log_times[
                        0]):
                log_times.pop(0)
                is_leader = elect_group.is_leader()
                leader = elect_group.get_leader()
                members = elect_group.get_members()
                log.append((is_leader, leader, members))
                default_logger.info(
                    "log: Process {}: leader={}, is_leader={}, members={}"
                    .format(
                        rank, leader, is_leader, members
                ))
        elect_group.stop()
        return log

    ########################################################################
    # Test for TestElectionGroupStableRpc under normal network condition
    ########################################################################
    @pytest.mark.repeat(10)
    def test_elect_normal(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker)
                                    )
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_elect)
        watch(rpc_mocker, processes, watcher)
        assert [r[0] for r in result] == [True, False, False]
        assert [r[1] for r in result] == [0, 0, 0]
        assert [r[2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    ########################################################################
    # Test for TestElectionGroupStableRpc where leader 0 is disconnected
    ########################################################################
    @pytest.mark.repeat(10)
    def test_elect_drop_leader(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        # drop the first leader 0
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker)
                                    )
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_elect,
                                    args_list=[(20,)] * 3)
        rpc_mocker.watch()

        begin = time.time()
        while True:
            for p in processes[0]:
                p.watch()
            if time.time() - begin >= self.TIMEOUT_DELTA * 9:
                Rpc.drop_match = lambda _, src, to: to == 0 or src == 0
            if watcher():
                break

        # because process 0 is disconnected from 1 and 2, it will still
        # consider itself the leader
        assert [r[0] for r in result] == [True, True, False]
        assert [r[1] for r in result] == [0, 1, 1]
        assert [r[2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    ########################################################################
    # Test for TestElectionGroupStableRpc where leader 0 is disconnected,
    # and the connection is recovered afterwards
    ########################################################################
    @pytest.mark.repeat(10)
    def test_elect_drop_leader_recover(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        # drop the first leader 0, then recover the links
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker)
                                    )
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_elect,
                                    args_list=[(30,)] * 3)
        rpc_mocker.watch()

        begin = time.time()
        # Warning! do not change the time configuration, it is used
        # to cover branches. (mainly about process 0 be notified of a
        # new leader 1)
        while True:
            for p in processes[0]:
                p.watch()
            if time.time() - begin >= self.TIMEOUT_DELTA * 9:
                Rpc.drop_match = lambda _, src, to: to == 0 or src == 0
            if time.time() - begin >= self.TIMEOUT_DELTA * 19:
                Rpc.drop_match = lambda _, src, to: False
            if watcher():
                break

        # process 0 will demote itself after connection reestablished
        assert [r[0] for r in result] == [False, True, False]
        assert [r[1] for r in result] == [1, 1, 1]
        assert [r[2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    ########################################################################
    # Test for TestElectionGroupStableRpc where leader 0 to 2 is disconnected
    # respectively, and then recovered afterwards
    ########################################################################
    @pytest.mark.repeat(10)
    def test_elect_multi_drop_recover(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        # drop the first leader 0
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker)
                                    )
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_elect_multi,
                                    args_list=[(50, (9, 19, 29, 39, 49))] * 3)
        rpc_mocker.watch()

        begin = time.time()
        while True:
            time.sleep(self.TIMEOUT_DELTA / 1000)
            for p in processes[0]:
                p.watch()
            if time.time() - begin >= self.TIMEOUT_DELTA * 10:
                Rpc.drop_match = lambda _, src, to: to == 0 or src == 0
            if time.time() - begin >= self.TIMEOUT_DELTA * 20:
                Rpc.drop_match = lambda _, src, to: to == 1 or src == 1
            if time.time() - begin >= self.TIMEOUT_DELTA * 30:
                Rpc.drop_match = lambda _, src, to: to == 2 or src == 2
            if time.time() - begin >= self.TIMEOUT_DELTA * 40:
                Rpc.drop_match = lambda *_: False
            if watcher():
                break

        # process 0 will demote itself after connection reestablished
        assert [r[0][0] for r in result] == [True, False, False]
        assert [r[0][1] for r in result] == [0, 0, 0]
        assert [r[0][2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

        assert [r[1][0] for r in result] == [True, True, False]
        assert [r[1][1] for r in result] == [0, 1, 1]
        assert [r[1][2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

        assert [r[2][0] for r in result] == [False, True, True]
        assert [r[2][1] for r in result] == [2, 1, 2]
        assert [r[2][2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

        assert [r[3][0] for r in result] == [True, False, True]
        assert [r[3][1] for r in result] == [0, 0, 2]
        assert [r[3][2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

        assert [r[4][0] for r in result] == [True, False, False]
        assert [r[4][1] for r in result] == [0, 0, 0]
        assert [r[4][2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    ########################################################################
    # Fuzzy testing for TestElectionGroupStableRpc where connections are
    # randomly lossy and randomly slow
    ########################################################################
    @pytest.mark.repeat(10)
    @pytest.mark.no_cover
    def test_elect_fuzzy(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        # drop the first leader 0
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker)
                                    )
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")

        # randomly select some time points to interrupt
        normal_time = [random.random() * 20 for _ in range(5)]
        interrupt_conns = [
            (random.sample([(0, 1), (1, 0),
                           (0, 2), (2, 0),
                           (1, 2), (2, 1)], k=random.randint(1, 5)),
             random.random() * 0.7 + 0.3)
            for _ in range(5)
        ]

        interrupt_time = [random.random() * 10 + 5 for _ in range(20)]

        total_time = 0
        log_time = []
        interrupt_periods = []
        normal_periods = []
        for nt, it in zip(normal_time, interrupt_time):
            period = nt + it + 10
            normal_periods.append((total_time, total_time + nt))
            interrupt_periods.append((total_time + nt, total_time + nt + it))
            normal_periods.append((total_time + nt + it, total_time + period))
            log_time.append(total_time + period)
            total_time += period

        total_time += 1

        default_logger.info("interrupt_periods:{}".format(interrupt_periods))
        default_logger.info("normal_periods:{}".format(normal_periods))
        default_logger.info("log_time:{}".format(log_time))

        result, watcher = run_multi(processes,
                                    self.subproc_test_elect_multi,
                                    args_list=[(total_time,
                                                log_time)] * 3)
        rpc_mocker.watch()
        begin = time.time()
        while True:
            time.sleep(self.TIMEOUT_DELTA / 1000)
            cur_time = time.time() - begin
            for p in processes[0]:
                p.watch()
            if (interrupt_periods and
                    self.TIMEOUT_DELTA * interrupt_periods[0][0] <= cur_time
                    < self.TIMEOUT_DELTA * interrupt_periods[0][1]):
                interrupt_conn = interrupt_conns[0]
                interrupt_periods.pop(0)
                interrupt_conns.pop(0)

                def drop_matcher(_, src, to):
                    return ((src, to) in interrupt_conn[0] and
                            random.random() > interrupt_conn[1])

                rpc_mocker.set_drop_match(drop_matcher)
            elif (normal_periods and
                    self.TIMEOUT_DELTA * normal_periods[0][0] <= cur_time
                    < self.TIMEOUT_DELTA * normal_periods[0][1]):
                normal_periods.pop(0)
                rpc_mocker.set_drop_match(None)

            if watcher():
                break

        for i in range(len(result)):
            is_leader = [r[i][0] for r in result]
            leader = [r[i][1] for r in result]
            assert len(set(leader)) == 1 and leader[0] in [0, 1, 2]
            assert is_leader[leader[0]]
            is_leader.pop(leader[0])
            assert not any(is_leader)

    @staticmethod
    def patch_and_init(rank, patches):
        init_rpc, shutdown, rpc_async, rpc_sync = dill.loads(patches)
        mock.patch("torch.distributed.rpc.init_rpc", init_rpc).start()
        mock.patch("torch.distributed.rpc.shutdown", shutdown).start()
        mock.patch("machin.parallel.distributed."
                   "election.rpc.rpc_async", rpc_async).start()
        mock.patch("machin.parallel.distributed."
                   "election.rpc.rpc_sync", rpc_sync).start()
        rpc.init_rpc(str(rank))

    @staticmethod
    def get_patches(rpc_mocker):
        return [(dill.dumps((
            rpc_mocker.get_mocker_init_rpc(i),
            rpc_mocker.get_mocker_shutdown(),
            rpc_mocker.get_mocker_rpc_async(),
            rpc_mocker.get_mocker_rpc_sync()
        )),) for i in [0, 1, 2]]
