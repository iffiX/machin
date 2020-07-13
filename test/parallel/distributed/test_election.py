from machin.parallel.distributed import (
    ElectionGroupSimple,
    ElectionGroupStableRpc
)
from machin.utils.logging import INFO
from ..util_run_multi import *
import time
import pytest
import random


class TestElectionGroupSimple(object):
    ########################################################################
    # Test for ElectionGroupSimple.get_leader
    ########################################################################
    @staticmethod
    def subproc_test_get_leader(rank):
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


class TestElectionGroupStableRpc(RpcTestBase):
    # Stable Election Algorithm is first tested under the normal condition
    # with no lossy / slow rpc links,
    # and then tested with the fuzzy testing method
    TIMEOUT_DELTA = 5e-1

    ########################################################################
    # Test routine for sub processes
    ########################################################################
    @classmethod
    def subproc_test_elect(cls, rank, run_time=9):
        # election function for all tests
        elect_group = ElectionGroupStableRpc(name="elect_group",
                                             member_ranks=[0, 1, 2],
                                             rank=rank,
                                             timeout=cls.TIMEOUT_DELTA,
                                             logging=True)
        elect_group.logger.setLevel(INFO)
        default_logger.info("Start election group on {}".format(rank))
        elect_group.start()
        time.sleep(cls.TIMEOUT_DELTA * run_time)
        elect_group.watch()
        is_leader = elect_group.is_leader()
        leader = elect_group.get_leader()
        members = elect_group.get_members()
        elect_group.stop()
        rpc.shutdown()
        return is_leader, leader, members

    ########################################################################
    # Test routine for sub processes, can collect multiple samples
    ########################################################################
    @classmethod
    def subproc_test_elect_multi(cls, rank, run_time=9, log_times=()):
        # election function for all tests
        elect_group = ElectionGroupStableRpc(name="elect_group",
                                             member_ranks=[0, 1, 2],
                                             rank=rank,
                                             timeout=cls.TIMEOUT_DELTA,
                                             logging=True)
        elect_group.logger.setLevel(INFO)
        default_logger.info("Start election group on {}".format(rank))

        begin = time.time()
        elect_group.start()

        log = []
        log_times = list(log_times)
        while time.time() - begin < cls.TIMEOUT_DELTA * run_time:
            time.sleep(cls.TIMEOUT_DELTA / 100)
            elect_group.watch()
            if (log_times and
                    time.time() - begin >=
                    cls.TIMEOUT_DELTA * log_times[0]):
                log_times.pop(0)
                is_leader = elect_group.is_leader()
                leader = elect_group.get_leader()
                members = elect_group.get_members()
                log.append((is_leader, leader, members))
                default_logger.info(
                    "log: Process {}: leader={}, is_leader={}, members={}"
                    .format(
                        rank, leader, is_leader, members
                    )
                )
        elect_group.stop()
        rpc.shutdown()
        return log

    ########################################################################
    # Test for ElectionGroupStableRpc under normal network condition
    ########################################################################
    @pytest.mark.repeat(10)
    def test_elect_normal(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker))
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_elect)
        watch(rpc_mocker, processes, watcher)
        assert [r[0] for r in result] == [True, False, False]
        assert [r[1] for r in result] == [0, 0, 0]
        assert [r[2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    ########################################################################
    # Test for ElectionGroupStableRpc where leader 0 is disconnected
    ########################################################################
    @pytest.mark.repeat(10)
    def test_elect_drop_leader(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        # drop the first leader 0
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker))
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_elect,
                                    args_list=[(20,)] * 3)
        rpc_mocker.watch()

        begin = time.time()
        last_report = -1
        while True:
            for p in processes[0]:
                p.watch()
            if time.time() - begin >= self.TIMEOUT_DELTA * 9:
                Rpc.drop_match = lambda _, src, to: to == 0 or src == 0

            last_report = self.report_time(last_report, begin)
            if watcher():
                break

        # because process 0 is disconnected from 1 and 2, it will still
        # consider itself the leader
        assert [r[0] for r in result] == [True, True, False]
        assert [r[1] for r in result] == [0, 1, 1]
        assert [r[2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    ########################################################################
    # Test for ElectionGroupStableRpc where leader 0 is disconnected,
    # and the connection is recovered afterwards
    ########################################################################
    @pytest.mark.repeat(10)
    def test_elect_drop_leader_recover(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        # drop the first leader 0, then recover the links
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker))
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_elect,
                                    args_list=[(30,)] * 3)
        rpc_mocker.watch()

        begin = time.time()
        last_report = -1
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

            last_report = self.report_time(last_report, begin)
            if watcher():
                break

        # process 0 will demote itself after connection reestablished
        assert [r[0] for r in result] == [False, True, False]
        assert [r[1] for r in result] == [1, 1, 1]
        assert [r[2] for r in result] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    ########################################################################
    # Test for ElectionGroupStableRpc where leader 0 to 2 is disconnected
    # respectively, and then recovered afterwards
    ########################################################################
    @pytest.mark.repeat(10)
    def test_elect_multi_drop_recover(self, processes, rpc_mocker):
        rpc_mocker.set_drop_match(None)
        # drop the first leader 0
        result, watcher = run_multi(processes,
                                    self.patch_and_init,
                                    args_list=self.get_patches(rpc_mocker))
        watch(rpc_mocker, processes, watcher)
        default_logger.info("All rpc inited")
        result, watcher = run_multi(processes,
                                    self.subproc_test_elect_multi,
                                    args_list=[(50, (9, 19, 29, 39, 49))] * 3)
        rpc_mocker.watch()

        begin = time.time()
        last_report = -1
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

            last_report = self.report_time(last_report, begin)
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
                                    args_list=self.get_patches(rpc_mocker))
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
        last_report = -1
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

            last_report = self.report_time(last_report, begin)
            if watcher():
                break

        for i in range(len(result)):
            is_leader = [r[i][0] for r in result]
            leader = [r[i][1] for r in result]
            assert len(set(leader)) == 1 and leader[0] in [0, 1, 2]
            assert is_leader[leader[0]]
            is_leader.pop(leader[0])
            assert not any(is_leader)

    def report_time(self, last_report, begin):
        cur_time = (time.time() - begin) / self.TIMEOUT_DELTA
        diff_round = abs(round(cur_time) - cur_time)
        if last_report < round(cur_time) and diff_round < 0.1:
            last_report = round(cur_time)
            default_logger.critical("Time: {}".format(last_report))
        return last_report
