from machin.parallel.distributed import World
from machin.parallel.process import Process
from machin.utils.logging import default_logger
import pytest
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


class TestWorld(object):
    ########################################################################
    # Test routine for sub processes
    ########################################################################
    @classmethod
    def subproc_test_elect(cls, rank):
        # election function for all tests
        global world
        world = ElectionGroupStableRpc(name="elect_group",
                                             member_ranks=[0, 1, 2],
                                             rank=rank,
                                             leader_rank=0,
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
        return is_leader, leader, members