from time import sleep, time
from decorator import FunctionMaker
from logging.handlers import QueueHandler, QueueListener
from typing import List, Tuple, Dict, Any
from machin.parallel.distributed import World, get_world as gw
from machin.parallel.process import Process, ProcessException
import sys
import dill
import pytest
import itertools
import logging
import multiprocessing as mp
import socket
from contextlib import closing

get_world = gw
# use queue handler
default_logger = logging.getLogger("multi_default_logger")
default_logger.setLevel(logging.INFO)


class SafeExit(Exception):
    pass


def find_free_port():
    # this function is used to find a free port
    # since we are using the host network in docker
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def process_main(pipe, log_queue):
    handler = logging.handlers.QueueHandler(log_queue)
    default_logger.addHandler(handler)
    while True:
        func, args, kwargs = dill.loads(pipe.recv())
        pipe.send(func(*args, **kwargs))


@pytest.fixture(scope="function")
def processes():
    ctx = mp.get_context("spawn")
    pipes = [mp.Pipe(duplex=True) for _ in [0, 1, 2]]
    man = ctx.Manager()
    queue = man.Queue()
    processes = [
        Process(target=process_main, args=(pipes[i][0], queue), ctx=ctx)
        for i in [0, 1, 2]
    ]

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] <%(levelname)s>:%(name)s:%(message)s")
    )

    ql = QueueListener(queue, handler)
    ql.start()
    default_logger.addHandler(handler)

    for p, i in zip(processes, [0, 1, 2]):
        default_logger.info(f"processes {i} started")
        p.start()
    yield processes, [pi[1] for pi in pipes]
    for p, pi, i in zip(processes, pipes, [0, 1, 2]):
        # try graceful shutdown first
        pi[1].send(dill.dumps((exit, 0, {})))
        p.join(timeout=1)
        if p.is_alive():
            # ungraceful shutdown
            default_logger.info(f"processes {i} ungraceful shutdown")
            p.terminate()
            p.join()
    default_logger.removeHandler(handler)
    ql.stop()
    man.shutdown()
    man.join()
    default_logger.info("processes stopped")


def exec_with_process(
    processes, func, args_list, kwargs_list, expected_results, timeout, *pass_through
):
    procs, proc_pipes = processes
    args_list = args_list if args_list is not None else itertools.repeat([])
    kwargs_list = kwargs_list if kwargs_list is not None else itertools.repeat({})

    # possibility of port collision using this method still exists
    port = find_free_port()
    for pi, rank, args, kwargs in zip(proc_pipes, [0, 1, 2], args_list, kwargs_list):
        kwargs["_world_port"] = port
        pi.send(dill.dumps((func, [rank] + list(args) + list(pass_through), kwargs)))

    results = [None, None, None]
    finished = [False, False, False]

    begin = time()
    while not all(finished):
        if time() - begin >= timeout:
            raise TimeoutError("Run-multi timeout!")
        for p, pi, i in zip(procs, proc_pipes, [0, 1, 2]):
            try:
                p.watch()
            except ProcessException as e:
                if "SafeExit" in e.args[0]:
                    return
                else:
                    raise e
            if pi.poll(timeout=1e-1):
                results[i] = pi.recv()
                finished[i] = True
        sleep(1e-1)
    if expected_results is not None:
        assert results == expected_results


def run_multi(
    args_list: List[Tuple[Any]] = None,
    kwargs_list: List[Dict[str, Any]] = None,
    expected_results: List[Any] = None,
    pass_through: List[str] = None,
    timeout: int = 60,
):
    # pass_through allows you to pass through pytest parameters and fixtures
    # to the sub processes, these pass through parameters must be placed
    # behind normal args and before kwargs
    assert args_list is None or len(args_list) == 3
    assert kwargs_list is None or len(kwargs_list) == 3
    assert expected_results is None or len(expected_results) == 3

    if pass_through is None:
        pt_args = ""
    else:
        pt_args = "," + ",".join(pass_through)

    def deco(func):
        return FunctionMaker.create(
            f"w_wrapped_func(processes{pt_args})",
            f"""
            return exec_with_process(
                processes, func, args_list, kwargs_list, 
                expected_results, timeout{pt_args})
            """,
            dict(
                args_list=args_list,
                kwargs_list=kwargs_list,
                expected_results=expected_results,
                timeout=timeout,
                func=func,
                exec_with_process=exec_with_process,
            ),
        )

    return deco


class WorldTestBase:
    @staticmethod
    def setup_world(func):
        def wrapped(rank, *args, _world_port=9100, **kwargs):
            # election function for all tests
            world = World(world_size=3, rank=rank, name=str(rank))
            default_logger.info(f"World using port {_world_port}")
            # set a temporary success attribute on world
            default_logger.info(f"World created on {rank}")
            result = func(rank, *args, **kwargs)
            world.stop()
            default_logger.info(f"World stopped on {rank}")
            return result

        return wrapped
