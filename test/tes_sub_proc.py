from machin.parallel.distributed import (
    ElectionGroupSimple,
)

import pytest
import itertools
import multiprocessing as mp

# process number is 3
elect_group = None


def process_main(pipe):
    while True:
        func, args, kwargs = pipe.recv()
        pipe.send(func(*args, **kwargs))


def processes():
    mp.set_start_method('spawn')
    pipes = [mp.Pipe(duplex=True) for _ in [0, 1, 2]]
    processes = [mp.Process(target=process_main,
                            args=(pipes[i][0],))
                 for i in [0, 1, 2]]
    for p in processes:
        p.start()
    print("created")
    yield [pi[1] for pi in pipes]
    print("teardown")
    for p, pi in zip(processes, pipes):
        pi[1].send((exit, (0,), {}))
        p.join()
        assert not p.exitcode


def run_with_mutli_proc(proc_pipes, func, args_list=None, kwargs_list=None):
    args_list = (args_list
                 if args_list is not None
                 else itertools.repeat([]))
    kwargs_list = (kwargs_list
                   if kwargs_list is not None
                   else itertools.repeat({}))
    for p, rank, args, kwargs in zip(proc_pipes, [0, 1, 2],
                                     args_list, kwargs_list):
        p.send((func, [rank] + list(args), kwargs))
    result = []
    for p in proc_pipes:
        result.append(p.recv())
    return result


class TestElectionGroupSimple:
    @staticmethod
    def subproc_test_get_leader(rank):
        global elect_group
        elect_group = ElectionGroupSimple(member_ranks=[0, 1, 2],
                                          rank=rank,
                                          leader_rank=0)
        elect_group.start()
        return elect_group.get_leader()

    def test_get_leader(self, processes):
        assert (run_with_mutli_proc(processes,
                                    self.subproc_test_get_leader)
                == [0, 0, 0])
        print("ok")


if __name__ == "__main__":
    gen = processes()
    TestElectionGroupSimple().test_get_leader(next(gen))
    try:
        next(gen)
    except StopIteration:
        exit(0)
