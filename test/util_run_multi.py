from machin.parallel.distributed import World, get_world
from machin.parallel.process import Process
from machin.utils.logging import default_logger
from time import sleep, time
import dill
import pytest
import itertools
import multiprocessing as mp


def process_main(pipe):
    while True:
        func, args, kwargs = dill.loads(pipe.recv())
        pipe.send(func(*args, **kwargs))


@pytest.fixture(scope="function")
def processes():
    ctx = mp.get_context("spawn")
    pipes = [mp.Pipe(duplex=True) for _ in [0, 1, 2]]
    processes = [Process(target=process_main, args=(pipes[i][0],), ctx=ctx)
                 for i in [0, 1, 2]]
    for p in processes:
        p.start()
    yield processes, [pi[1] for pi in pipes]
    for p, pi, i in zip(processes, pipes, [0, 1, 2]):
        # try graceful shutdown first
        pi[1].send(dill.dumps((exit, 0, {})))
        p.join(timeout=1)
        if p.is_alive():
            # ungraceful shutdown
            default_logger.info("processes {} ungraceful shutdown".format(i))
            p.terminate()
            p.join()
    default_logger.info("processes stopped")


def run_multi(args_list=None, kwargs_list=None, expected_results=None,
              timeout=10):
    assert len(expected_results) == 3

    def deco(func):
        def wrapped(processes):
            nonlocal args_list, kwargs_list, expected_results
            procs, proc_pipes = processes
            args_list = (args_list
                         if args_list is not None
                         else itertools.repeat([]))
            kwargs_list = (kwargs_list
                           if kwargs_list is not None
                           else itertools.repeat({}))
            for pi, rank, args, kwargs in zip(proc_pipes, [0, 1, 2],
                                              args_list, kwargs_list):
                pi.send(dill.dumps((func, [rank] + list(args), kwargs)))

            results = [None, None, None]
            finished = [False, False, False]

            begin = time()
            while not all(finished):
                if time() - begin >= timeout:
                    raise TimeoutError("Run-multi timeout!")
                for p, pi, i in zip(procs, proc_pipes, [0, 1, 2]):
                    p.watch()
                    if pi.poll(timeout=1e-1):
                        results[i] = pi.recv()
                        finished[i] = True
                sleep(1e-1)
            if expected_results is not None:
                assert results == expected_results

        return wrapped
    return deco


class WorldTestBase(object):
    @staticmethod
    def setup_world(func):
        def wrapped(rank, *args, **kwargs):
            # election function for all tests
            world = World(world_size=3, rank=rank,
                          name=str(rank), rpc_timeout=1)

            # set a temporary success attribute on world
            default_logger.info("World created on {}".format(rank))
            result = func(rank, *args, **kwargs)
            world.stop()
            default_logger.info("World stopped on {}".format(rank))
            return result

        return wrapped
