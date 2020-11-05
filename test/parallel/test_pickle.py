from machin.parallel.pickle import dumps, loads
from machin.parallel.process import Process
import torch as t
from multiprocessing import Pipe, get_context


def subproc_test_dumps_copy_tensor(pipe):
    """
    Send a subproc to subprocess.

    Args:
        pipe: (todo): write your description
    """
    pipe.send(dumps(t.zeros([10]), copy_tensor=True))


def test_dumps_copy_tensor():
    """
    A copy of the copy of a tensor.

    Args:
    """
    pipe_0, pipe_1 = Pipe(duplex=True)
    ctx = get_context("spawn")
    process_0 = Process(target=subproc_test_dumps_copy_tensor,
                        args=(pipe_0,), ctx=ctx)
    process_0.start()
    while process_0.is_alive():
        process_0.watch()
    assert t.all(loads(pipe_1.recv()) == t.zeros([10]))
    process_0.join()


def subproc_test_dumps_not_copy_tensor(pipe):
    """
    Subproc a copy of a tensor.

    Args:
        pipe: (todo): write your description
    """
    tensor = t.zeros([10])
    tensor.share_memory_()
    pipe.send(dumps(tensor, copy_tensor=False))


def test_dumps_not_copy_tensor():
    """
    A copy of a tensor.

    Args:
    """
    pipe_0, pipe_1 = Pipe(duplex=True)
    ctx = get_context("fork")
    process_0 = Process(target=subproc_test_dumps_not_copy_tensor,
                        args=(pipe_0,), ctx=ctx)
    process_0.start()
    while process_0.is_alive():
        process_0.watch()
    assert t.all(loads(pipe_1.recv()) == t.zeros([10]))
    process_0.join()


def subproc_test_dumps_local_func(pipe):
    """
    Decor for subproc.

    Args:
        pipe: (todo): write your description
    """
    tensor = t.zeros([10])
    tensor.share_memory_()

    def local_func():
        """
        Returns a function that will be used function.

        Args:
        """
        nonlocal tensor
        return tensor

    pipe.send(dumps(local_func, copy_tensor=False))


def test_dumps_local_func():
    """
    A simple wrapper around the local process.

    Args:
    """
    pipe_0, pipe_1 = Pipe(duplex=True)
    ctx = get_context("fork")
    process_0 = Process(target=subproc_test_dumps_local_func,
                        args=(pipe_0,), ctx=ctx)
    process_0.start()
    while process_0.is_alive():
        process_0.watch()
    assert t.all(loads(pipe_1.recv())() == t.zeros([10]))
    process_0.join()
