from machin.parallel.process import Process
from machin.utils.logging import default_logger
from .util_rpc_mocker import RpcMocker
from torch.distributed import rpc
from typing import Any, List
from time import sleep
import dill
import mock
import pytest
import itertools
import multiprocessing as mp


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
                     print_expired=False, rpc_response_time=[1e-3, 1e-2])
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
            if pi.poll(timeout=1e-1):
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
        sleep(1e-1)


class Rpc(RpcMocker):
    drop_match = lambda *_: False

    @staticmethod
    def interposer(obj, timestamp, src, to, fuzz, token):
        cmd, obj, timestamp, src, to, fuzz, token = \
            RpcMocker.interposer(obj, timestamp, src, to, fuzz, token)
        if Rpc.drop_match(obj, src, to):
            cmd = "drop"
        args = obj[2]
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


class RpcTestBase(object):
    @staticmethod
    def patch_and_init(rank, patches):
        init_rpc, shutdown, rpc_async, rpc_sync = dill.loads(patches)
        mock.patch("torch.distributed.rpc.init_rpc", init_rpc).start()
        mock.patch("torch.distributed.rpc.shutdown", shutdown).start()
        mock.patch("machin.parallel.distributed."
                   "election.rpc.rpc_async", rpc_async).start()
        mock.patch("machin.parallel.distributed."
                   "election.rpc.rpc_sync", rpc_sync).start()
        mock.patch("machin.parallel.distributed."
                   "role_dispatcher.rpc.rpc_async", rpc_async).start()
        mock.patch("machin.parallel.distributed."
                   "role_dispatcher.rpc.rpc_sync", rpc_sync).start()
        rpc.init_rpc(str(rank))

    @staticmethod
    def get_patches(rpc_mocker):
        return [(dill.dumps((
            rpc_mocker.get_mocker_init_rpc(i),
            rpc_mocker.get_mocker_shutdown(),
            rpc_mocker.get_mocker_rpc_async(),
            rpc_mocker.get_mocker_rpc_sync()
        )),) for i in [0, 1, 2]]
