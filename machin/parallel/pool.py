import os
import sys
import time
import queue
import warnings
import itertools
import threading
import collections
from typing import Collection, Iterable, Callable, Union, Tuple, List, Dict, Any
from enum import Enum, unique
from multiprocessing import get_context, TimeoutError
from machin.utils.logging import default_logger

from .exception import ExceptionWithTraceback
from .pickle import dumps, loads
from .process import Process
from .thread import Thread
from .queue import SimpleQueue, MultiP2PQueue
from .util import Finalize


def map_caller(args):
    return list(map(*args))


def starmap_caller(args):
    return list(itertools.starmap(args[0], args[1]))


def proxy_caller(*input_):
    """
    Call a serialized function and return results.
    """
    if len(input_) == 1:
        func_str, args, kwargs = input_[0]
    else:
        func_str, args, kwargs = input_
    func = loads(func_str)
    return func(*args, **kwargs)


def proxy_ctx_caller(*input_):
    """
    Call a serialized function with worker context and return results.
    """
    if len(input_) == 1:
        func_str, args, kwargs = input_[0]
    else:
        func_str, args, kwargs = input_
    func = loads(func_str)
    return func(CtxPoolStorage.storage, *args, **kwargs)


def proxy_dumper(recurse, copy_tensor, func, args_list):
    """
    Serialize a function so it can be called.

    Returns:
        List[function string, arguments...]
    """
    # recurse will enable context variable saving
    dump = dumps(func, recurse=recurse, copy_tensor=copy_tensor)
    return [(dump, args, {}) for args in args_list]


@unique
class PoolStates(Enum):
    RUN = 0
    CLOSE = 1
    TERMINATE = 2


class AsyncResult:
    """
    Class whose instances are returned by `Pool.apply_async()`
    """

    def __init__(self, job, cache, callback, error_callback):
        self._event = threading.Event()
        self._job = job
        self._cache = cache
        self._callback = callback
        self._error_callback = error_callback
        self._success = False
        self._value = None
        cache[self._job] = self

    def ready(self) -> bool:
        """
        Return whether the call has completed.
        """
        return self._event.is_set()

    def successful(self) -> bool:
        """
        Return whether the call completed without raising an exception.

        Will raise `ValueError` if the result is not ready.
        """
        if not self.ready():
            raise ValueError(f"{self:0!r} not ready")
        return self._success

    def wait(self, timeout: float = None):
        """
        Wait until the result is available or until timeout seconds pass.

        Args:
            timeout: Timeout in seconds.
        """
        self._event.wait(timeout)

    def get(self, timeout=None) -> Any:
        """
        Return the result when it arrives.

        If timeout is not None and the result does not arrive within timeout seconds
        then `multiprocessing.TimeoutError` is raised.

        If the remote call raised an exception then that exception will be reraised
        by `get()`.

        Args:
            timeout: Timeout in seconds.

        Returns:
            The result.
        """
        self.wait(timeout)
        if not self.ready():
            raise TimeoutError
        if self._success:
            return self._value
        else:
            raise self._value

    def set(self, _chunk_idx, obj):
        """
        Called by the pool to set result.
        """
        self._success, self._value = obj
        if self._callback and self._success:
            self._callback(self._value)
        if self._error_callback and not self._success:
            self._error_callback(self._value)
        self._event.set()
        del self._cache[self._job]


class MapResult(AsyncResult):
    """
    Class whose instances are returned by `Pool.map_async()`
    """

    def __init__(self, job, cache, chunksize, length, callback, error_callback):
        super().__init__(job, cache, callback, error_callback)
        self._success = True
        self._value = [None] * length
        self._chunksize = chunksize
        if chunksize <= 0:
            self._number_left = 0
            self._event.set()
            del cache[self._job]
        else:
            # equal to ceil(length / chunk_size)
            self._number_left = length // chunksize + bool(length % chunksize)

    def set(self, chunk_idx, obj):
        """
        Called by the pool to set result.
        """
        self._number_left -= 1
        success, result = obj
        if success and self._success:
            self._value[
                chunk_idx * self._chunksize : (chunk_idx + 1) * self._chunksize
            ] = result
            if self._number_left == 0:
                if self._callback:
                    self._callback(self._value)
                del self._cache[self._job]
                self._event.set()
        else:
            if not success and self._success:
                # only store first exception
                self._success = False
                self._value = result
            if self._number_left == 0:
                # only consider the result ready once all jobs are done
                if self._error_callback:
                    self._error_callback(self._value)
                del self._cache[self._job]
                self._event.set()


class IMapIterator:
    """
    Class whose instances are returned by `Pool.imap()`
    """

    def __init__(self, job, cache, length):
        self._cond = threading.Condition(threading.Lock())
        self._job = job
        self._cache = cache
        self._items = collections.deque()
        self._index = 0
        self._length = length
        self._unsorted = {}
        cache[self._job] = self

    def next(self, timeout=None) -> Any:
        """
        Return the next result within timeout.

        If timeout is reached and no new item is returned by the worker,
        and returned total item number is smaller than the job size,
        then raise an `TimeoutError`.

        If total item number is equal than the job size (all jobs finished and
        returned), then raise an `StopIteration`.

        Args:
            timeout: Timeout in seconds.
        """
        with self._cond:
            try:
                item = self._items.popleft()
            except IndexError:
                if self._index == self._length:
                    raise StopIteration from None
                self._cond.wait(timeout)
                try:
                    item = self._items.popleft()
                except IndexError:
                    if self._index == self._length:
                        raise StopIteration from None
                    raise TimeoutError from None

        success, value = item
        if success:
            return value
        raise value

    def set(self, chunk_idx, obj):
        """
        Called by the pool to set result.
        """
        with self._cond:
            if self._index == chunk_idx:
                self._items.append(obj)
                self._index += 1
                # group items in unsorted map following the current _index
                # stop when a gap in _index is detected.
                while self._index in self._unsorted:
                    obj = self._unsorted.pop(self._index)
                    self._items.append(obj)
                    self._index += 1
                self._cond.notify()
            else:
                self._unsorted[chunk_idx] = obj

            if self._index == self._length:
                del self._cache[self._job]

    def __iter__(self):
        return self

    def __next__(self):
        # when users call iterator with `next(it)` and not providing the timeout
        return self.next()


class IMapUnorderedIterator(IMapIterator):
    """
    Class whose instances are returned by `Pool.imap_unordered()`
    """

    def set(self, _chunk_idx, obj):
        """
        Called by the pool to set result.
        """
        with self._cond:
            self._items.append(obj)
            self._index += 1
            self._cond.notify()
            if self._index == self._length:
                del self._cache[self._job]


class BasePool:
    """
    The basic pool class, adapted from python 3.7.3 multiprocessing.pool.

    Note:
        The exception thrown while iterating the iterable will not be
        reraised and will be thrown here. This is different from the
        original implementation.
    """

    def __init__(
        self,
        processes=None,
        initializer=None,
        initargs=(),
        maxtasksperchild=None,
        context=None,
    ):
        self._ctx = context or get_context()
        self._inqueue, self._outqueue = self.setup_queues()
        self._cache = {}
        self._state = PoolStates.RUN
        self._maxtasksperchild = maxtasksperchild
        self._initializer = initializer
        self._initargs = initargs
        self._job_counter = 0
        self._job_submit_lock = threading.Lock()

        if processes is None:
            processes = os.cpu_count() or 1
        if processes < 1:
            raise ValueError("Number of processes must be at least 1")

        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")

        self._processes = processes
        self._pool = []

        # create workers
        self.repopulate_pool()

        # create handler threads
        self._worker_handler = threading.Thread(
            target=BasePool._handle_workers, args=(self,)
        )
        self._worker_handler.daemon = True
        self._worker_handler.start()

        self._result_handler = threading.Thread(
            target=BasePool._handle_results, args=(self,)
        )
        self._result_handler.daemon = True
        self._result_handler.start()

        # create weakref finalizer
        self._terminate = Finalize(
            self,
            self._finalize_pool,
            args=(self, self._worker_handler, self._result_handler),
            exitpriority=15,
        )

    def apply(self, func: Callable, args: Tuple = (), kwds: Dict = None) -> Any:
        """
        Equivalent of `func(*args, **kwds)`.

        Args:
            func: Function to call.
            args: Arguments provided to the function call.
            kwds: Keyword arguments provided to the function call.

        Returns:
            Function call result.
        """
        return self.apply_async(func, args, kwds).get()

    def apply_async(
        self,
        func: Callable,
        args: Tuple = (),
        kwds: Dict = None,
        callback: Callable[[Any], None] = None,
        error_callback: Callable[[Exception], None] = None,
    ) -> AsyncResult:
        """
        Asynchronous version of `apply()` method.

        Args:
            func: Function to call.
            args: Arguments provided to the function call.
            kwds: Keyword arguments provided to the function call.
            callback: Callback function to apply on the result.
            error_callback: Error callback function to apply on the exception instance.

        Returns:
            An instance of ``AsyncResult``.
        """
        if self._state != PoolStates.RUN:
            raise ValueError("Pool not running")
        job_idx = self._next_job_idx()
        result = AsyncResult(job_idx, self._cache, callback, error_callback)
        self._submit_task((job_idx, 0, func, args, kwds or {}))
        return result

    def map(
        self,
        func: Callable[[Any], Any],
        iterable: Collection[Any],
        chunksize: int = None,
    ) -> List[Any]:
        """
        Apply `func` to each element in `iterable`, collecting the results
        in a list that is returned.

        Args:
            func: Function to call.
            iterable: A collection of single argument provided to the function call.
            chunksize: Size of iterable chunk assigned to each worker.

        Returns:
            A list of result from applying the function on each item in the iterable.
        """
        return self._map_async(func, iterable, map_caller, chunksize).get()

    def map_async(
        self,
        func: Callable[[Any], Any],
        iterable: Collection[Any],
        chunksize: int = None,
        callback: Callable[[Any], None] = None,
        error_callback: Callable[[Exception], None] = None,
    ) -> AsyncResult:
        """
        Asynchronous version of `map()` method.

        Args:
            func: Function to call.
            iterable: A collection of single argument provided to the function call.
            chunksize: Size of iterable chunk assigned to each worker.
            callback: Callback function to apply on the result.
            error_callback: Error callback function to apply on the exception instance.

        Returns:
            An instance of ``AsyncResult``.
        """
        return self._map_async(
            func, iterable, map_caller, chunksize, callback, error_callback
        )

    def starmap(
        self,
        func: Callable[[Any], Any],
        iterable: Collection[Tuple],
        chunksize: int = None,
    ) -> List[Any]:
        """
        Like `map()` method but the elements of the `iterable` are expected to
        be iterables as well and will be unpacked as arguments. Hence
        `func` and (a, b) becomes func(a, b).

        Args:
            func: Function to call.
            iterable: A collection of tuples of arguments provided to the function call.
            chunksize: Size of iterable chunk assigned to each worker.

        Returns:
            A list of result from applying the function on each tuple in the iterable.
        """
        return self._map_async(func, iterable, starmap_caller, chunksize).get()

    def starmap_async(
        self,
        func: Callable[[Any], Any],
        iterable: Collection[Tuple],
        chunksize: int = None,
        callback: Callable[[Any], None] = None,
        error_callback: Callable[[Exception], None] = None,
    ) -> AsyncResult:
        """
        Asynchronous version of `starmap()` method.

        Args:
            func: Function to call.
            iterable: A collection of tuples of arguments provided to the function call.
            chunksize: Size of iterable chunk assigned to each worker.
            callback: Callback function to apply on the result.
            error_callback: Error callback function to apply on the exception instance.

        Returns:
            An instance of ``AsyncResult``.
        """
        return self._map_async(
            func, iterable, starmap_caller, chunksize, callback, error_callback
        )

    def imap(
        self, func: Callable[[Any], Any], iterable: Collection[Any], chunksize: int = 1,
    ) -> Union[IMapIterator, List[Any]]:
        """
        Equivalent of `map()`, but will not store all results, instead, get one
        at a time in the sequential order.

        Args:
            func: Function to call.
            iterable: A collection of single argument provided to the function call.
            chunksize: Size of iterable chunk assigned to each worker.

        Returns:
            ``ImapIterator`` when chunksize is set to 1, else a list of results.
        """
        return self._imap(func, iterable, IMapIterator, chunksize)

    def imap_unordered(
        self, func: Callable[[Any], Any], iterable: Collection[Any], chunksize: int = 1,
    ) -> Union[IMapUnorderedIterator, List[Any]]:
        """
        Like `imap()` method but ordering of results is arbitrary.

        Args:
            func: Function to call.
            iterable: A collection of single argument provided to the function call.
            chunksize: Size of iterable chunk assigned to each worker.

        Returns:
            ``ImapIterator`` when chunksize is set to 1, else a list of results.
        """
        return self._imap(func, iterable, IMapUnorderedIterator, chunksize)

    def _map_async(
        self, func, iterable, mapper, chunksize=None, callback=None, error_callback=None
    ):
        """
        Helper function to implement map, starmap and their async counterparts.
        """
        if self._state != PoolStates.RUN:
            raise ValueError("Pool not running")
        if not hasattr(iterable, "__len__"):
            iterable = list(iterable)

        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool) * 4)
            if extra:
                chunksize += 1
        if len(iterable) == 0:
            chunksize = 0

        job_idx = self._next_job_idx()
        task_batches = self._split_tasks(func, iterable, chunksize)
        result = MapResult(
            job_idx, self._cache, chunksize, len(iterable), callback, error_callback,
        )
        for chunk_idx, batch in enumerate(task_batches):
            self._submit_task((job_idx, chunk_idx, mapper, (batch,), {}))
        return result

    def _imap(
        self,
        func: Callable[[Any], Any],
        iterable: Collection[Tuple],
        iterator_class,
        chunksize: int = 1,
    ):
        """
        Helper function to implement imap and imap_unordered.
        """
        if self._state != PoolStates.RUN:
            raise ValueError("Pool not running")
        if not hasattr(iterable, "__len__"):
            iterable = list(iterable)

        job_idx = self._next_job_idx()
        if chunksize == 1:
            result = iterator_class(job_idx, self._cache, len(iterable))
            for chunk_idx, arg in enumerate(iterable):
                self._submit_task((job_idx, chunk_idx, func, (arg,), {}))
            return result
        else:
            if chunksize < 1:
                raise ValueError(f"Chunksize must be 1+, not {chunksize:n}")
            task_batches = self._split_tasks(func, iterable, chunksize)
            result = iterator_class(job_idx, self._cache, len(task_batches))
            for chunk_idx, batch in enumerate(task_batches):
                self._submit_task((job_idx, chunk_idx, map_caller, (batch,), {}))
            return [item for chunk in result for item in chunk]

    def close(self):
        """
        Softly closing the pool, handler threads, and then
        shutdown workers by sending signals. The pool will
        be closed after all job is finished and all results
        returned.

        Remember to call ``join()`` to wait for full shutdown.
        """
        default_logger.debug("Closing pool")
        if self._state == PoolStates.RUN:
            self._state = PoolStates.CLOSE

    def terminate(self):
        """
        Immediately terminates the pool threads and workers, and
        also join them.
        """
        default_logger.debug("Terminating pool")
        self._state = PoolStates.TERMINATE
        self._terminate()
        default_logger.debug("Terminating finished")

    def join(self):
        """
        Wait for handler threads and workers to join.
        """
        default_logger.debug("Joining pool")
        if self._state == PoolStates.RUN:
            raise ValueError("Pool is still running")
        self._worker_handler.join()
        self._result_handler.join()
        self.join_workers()
        default_logger.debug("Joining finished")

    def size(self) -> int:
        """
        Returns:
            The number of workers in pool.
        """
        return len(self._pool)

    # Begin overridable section
    def repopulate_pool(self):
        """
        Bring the number of pool workers up to the specified number,
        it also creates new workers to replace old workers which have
        exited after executing ``maxtasksperchild``.

        Override this method to implement your own pool.
        """
        for i in range(self._processes - len(self._pool)):
            w = Process(
                target=self.worker,
                args=(
                    self._inqueue.get,
                    self._outqueue.put,
                    self._initializer,
                    self._initargs,
                    self._maxtasksperchild,
                ),
                ctx=self._ctx,
            )
            self._pool.append(w)
            w.name = w.name.replace("Process", "PoolWorker")
            w.daemon = True
            w.start()
            default_logger.debug(f"Added worker {w.name}")

    def maintain_pool(self):
        """
        Watch workers for exceptions and raise them and then terminate the pool,
        Clean up any retired workers reaching max task number, and
        start replacements for them.

        Override this method to implement your own pool.
        """
        for i in reversed(range(len(self._pool))):
            worker = self._pool[i]
            if worker.exception is not None:
                default_logger.critical(worker.exception, exc_info=True)

            if worker.exitcode is not None:
                # worker exited
                default_logger.debug(
                    f"Cleaning up worker {worker.name}, " f"exitcode={worker.exitcode}"
                )
                worker.join()
                del self._pool[i]
        self.repopulate_pool()

    def terminate_workers(self):
        """
        Force terminate all workers.

        Override this method to implement your own pool.
        """
        for p in self._pool:
            if p.exitcode is None:
                p.terminate()
                default_logger.debug(f"Terminated worker {p.pid}")

    def join_workers(self):
        """
        Wait until all workers have terminated.

        Override this method to implement your own pool.
        """
        for p in self._pool:
            if p.is_alive():
                # worker has not yet exited
                p.join()
                default_logger.debug(f"Joined worker {p.pid}")

    def setup_queues(self):
        """
        Create an input queue and an output queue which will be used to communicate
        with workers.

        Override this method to implement your own pool.
        """
        return SimpleQueue(ctx=self._ctx), SimpleQueue(ctx=self._ctx)

    def pool_inqueue_put(self, obj: Any):
        """
        Put a task item into the input queue on the pool side. Note all

        Override this method to implement your own pool.
        """
        return self._inqueue.quick_put(obj)

    def pool_outqueue_get(self, timeout: float):
        """
        Read a result item from the output queue on the pool side.

        The method should block for timeout seconds, and then throw
        a ``TimeoutError`` if no result is available. It should also
        throw ``OSError`` or ``EOFError`` to indicate that it is
        improperly closed and cannot be used.


        Override this method to implement your own pool.
        """
        return self._outqueue.quick_get(timeout=timeout)

    @staticmethod
    def worker(
        get,
        put,
        initializer: Callable = None,
        initargs: Tuple = (),
        maxtasks: int = None,
    ):
        """
        The default worker function executed by worker processes.

        Override this method to implement your own pool.

        Args:
            get: A function of form ``get() -> Any`` used to get tasks.
            put: A function of form ``put(obj: Any)`` used to put results.
            initializer: An initializer function to init global environment in worker
                processes.
            initargs: Initializer arguments.
            maxtasks: Maximum number of tasks a worker needs to run before it exits.
        """
        if (maxtasks is not None) and not (isinstance(maxtasks, int) and maxtasks >= 1):
            raise AssertionError(f"Maxtasks {maxtasks:!r} is not valid")

        if initializer is not None:
            initializer(*initargs)

        completed = 0
        while maxtasks is None or (maxtasks and completed < maxtasks):
            task = get()

            if task is None:
                default_logger.debug("Worker got sentinel -- exiting")
                break

            # Job index is the index of the submitted batch of tasks.
            # Chunk index is the index of the chunk got by the worker.
            job_idx, chunk_idx, func, args, kwds = task
            try:
                result = (True, func(*args, **kwds))
            except Exception as e:
                result = (False, ExceptionWithTraceback(e))
            put((job_idx, chunk_idx, result))
            completed += 1
        default_logger.debug(f"Worker exiting after {completed} tasks")

    # End overridable section

    def _next_job_idx(self):
        job_idx = self._job_counter
        self._job_counter += 1
        return job_idx

    def _submit_task(self, task):
        with self._job_submit_lock:
            try:
                self.pool_inqueue_put(task)
            except Exception as e:
                job, idx = task[:2]
                try:
                    # an error occurred while putting task in queue
                    # set chunk result as exception
                    self._cache[job].set(idx, (False, e))
                except KeyError:
                    pass

    @staticmethod
    def _handle_workers(pool: "BasePool"):
        """
        Worker handler. Keep maintaining workers until the cache gets drained,
        unless the pool is terminated.
        """
        while pool._state == PoolStates.RUN or (
            pool._cache and pool._state != PoolStates.TERMINATE
        ):
            pool.maintain_pool()
            time.sleep(0.1)

        for _ in pool._pool:
            # send stop signals to workers
            pool.pool_inqueue_put(None)

        default_logger.debug("Worker handler exiting")

    @staticmethod
    def _handle_results(pool: "BasePool"):
        while pool._state == PoolStates.RUN or (
            pool._cache and pool._state != PoolStates.TERMINATE
        ):
            try:
                result = pool.pool_outqueue_get(0.1)
            except (OSError, EOFError) as e:
                default_logger.debug("Result handler got EOFError/OSError -- exiting")
                default_logger.critical(e, exc_info=True)
                return
            except TimeoutError:
                continue

            job_idx, chunk_idx, obj = result
            try:
                pool._cache[job_idx].set(chunk_idx, obj)
            except KeyError:
                pass

        default_logger.debug("Result handler exiting")

    @staticmethod
    def _split_tasks(func: Callable, it: Iterable, chunksize: int):
        """
        Create task batches of form::
            [(func, Tuple), (func, Tuple), ...]

        Where each tuple is a slice of ``chunk_size`` from ``it``.
        """
        it = iter(it)
        result = []
        while 1:
            # move iterator forward and get next slice of chunk_size
            x = tuple(itertools.islice(it, chunksize))
            if not x:
                return result
            result.append((func, x))

    @classmethod
    def _finalize_pool(cls, pool, worker_handler, result_handler):
        """
        Pool finalizer callback use by the Finalizer to clean up things using weakref.
        """
        # this is guaranteed to only be called once
        default_logger.debug("Finalizing pool")
        pool._state = PoolStates.TERMINATE

        # We must wait for the worker handler to exit before terminating
        # workers because we don't want workers to be restarted behind our back.
        default_logger.debug("Joining worker handler")
        if threading.current_thread() is not worker_handler:
            worker_handler.join()

        default_logger.debug("Joining result handler")
        if threading.current_thread() is not result_handler:
            result_handler.join()

        # Terminate workers which haven't already finished.
        default_logger.debug("Terminating workers")
        pool.terminate_workers()

        default_logger.debug("Joining pool workers")
        pool.join_workers()

    def __reduce__(self):
        raise NotImplementedError(
            "Pool objects cannot be passed between processes or pickled"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()


class Pool(BasePool):
    """
    Enhanced multiprocessing pool for pytorch, provides:
     1. Support for lambdas and local functions.
     2. Ability to select the tensor serialize scheme.
    """

    def __init__(
        self,
        processes=None,
        initializer=None,
        initargs=(),
        maxtasksperchild=None,
        is_recursive=False,
        is_daemon=True,
        is_copy_tensor=True,
        share_method=None,
    ):
        """
        Note:
            To share "cpu" tensors in shared memory, you must set::

                is_copy_tensor=False,
                share_method="cpu"

            To share "cuda" tensors, you must set::

                is_copy_tensor=False,
                share_method="cuda"

        Note:
            The default context used in pool is "spawn", to avoid any issues
            brought by "fork". "fork" will only be used if you want to pass
            cpu tensors in shared memory.

        Args:
            processes: Number of processes in the pool.
            initializer: Initializer function executed by the pool/
            initargs: Args passed to the init function.
            maxtasksperchild: Maximum number of tasks per worker process.
            is_recursive: Set to ``True`` to support local functions
                and lambdas.
            is_daemon: Whether worker processes in the pool are started as
                daemon processes.
            is_copy_tensor: Whether to copy tensors or pass tensors by
                reference to worker processes.
            share_method: If ``is_copy_tensor`` is ``False``, you must
                specify this argument. "cpu" means you may use cpu tensors
                in the shared memory, "cuda" means cuda tensors, you can only
                specify one share method.
        """
        if processes is None:
            processes = os.cpu_count() or 1
        if processes < 1:
            raise ValueError("Number of processes must be at least 1")

        context = get_context("spawn")
        if sys.platform.startswith("linux") and not is_copy_tensor:
            if share_method not in ("cpu", "cuda"):
                raise RuntimeError(f'Invalid share method: "{share_method}"')
            if share_method == "cpu":
                context = get_context("fork")
        else:
            warnings.warn(
                "Sharing but not copying a tensor is not supported "
                "on platforms other than linux."
            )
            is_copy_tensor = True

        self._ctx = context
        self._processes = processes

        self._is_recursive = is_recursive
        self._is_daemon = is_daemon
        self._is_copy_tensor = is_copy_tensor
        self._caller = proxy_caller

        super().__init__(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild,
            context=context,
        )

    def apply(self, func, args=(), kwds=None):
        # DOC INHERITED
        if kwds is None:
            kwds = {}
        return super().apply(
            self._caller,
            (
                (
                    dumps(
                        func,
                        recurse=self._is_recursive,
                        copy_tensor=self._is_copy_tensor,
                    ),
                    args,
                    kwds,
                ),
            ),
        )

    def apply_async(self, func, args=(), kwds=None, callback=None, error_callback=None):
        # DOC INHERITED
        if kwds is None:
            kwds = {}
        return super().apply_async(
            self._caller,
            (
                (
                    dumps(
                        func,
                        recurse=self._is_recursive,
                        copy_tensor=self._is_copy_tensor,
                    ),
                    args,
                    kwds,
                )
            ),
        )

    def map(self, func, iterable, chunksize=None):
        # DOC INHERITED
        return super().map(
            self._caller,
            proxy_dumper(
                self._is_recursive,
                self._is_copy_tensor,
                func,
                [(arg,) for arg in iterable],
            ),
            chunksize,
        )

    def map_async(
        self, func, iterable, chunksize=None, callback=None, error_callback=None
    ):
        # DOC INHERITED
        return super().map_async(
            self._caller,
            proxy_dumper(
                self._is_recursive,
                self._is_copy_tensor,
                func,
                [(arg,) for arg in iterable],
            ),
            chunksize,
            callback,
            error_callback,
        )

    def imap(self, func, iterable, chunksize=1):
        # DOC INHERITED
        return super().imap(
            self._caller,
            proxy_dumper(
                self._is_recursive,
                self._is_copy_tensor,
                func,
                [(arg,) for arg in iterable],
            ),
            chunksize,
        )

    def imap_unordered(self, func, iterable, chunksize=1):
        # DOC INHERITED
        return super().imap_unordered(
            self._caller,
            proxy_dumper(
                self._is_recursive,
                self._is_copy_tensor,
                func,
                [(arg,) for arg in iterable],
            ),
            chunksize,
        )

    def starmap(self, func, iterable, chunksize=None):
        # DOC INHERITED
        return super().starmap(
            self._caller,
            proxy_dumper(self._is_recursive, self._is_copy_tensor, func, iterable),
            chunksize,
        )

    def starmap_async(
        self, func, iterable, chunksize=None, callback=None, error_callback=None
    ):
        # DOC INHERITED
        return super().starmap_async(
            self._caller,
            proxy_dumper(self._is_recursive, self._is_copy_tensor, func, iterable),
            chunksize,
            callback,
            error_callback,
        )

    @staticmethod
    def worker(*args, **kwargs):
        import gc

        BasePool.worker(*args, **kwargs)
        # Regular multiprocessing workers don't fully clean up after themselves,
        # so we have to explicitly trigger garbage collection to make sure that all
        # destructors are called...
        gc.collect()

    def __reduce__(self):
        raise NotImplementedError(
            "Pool objects cannot be passed between processes or pickled"
        )


class P2PPool(Pool):
    def setup_queues(self):
        # queues are only used to send dumped strings
        return MultiP2PQueue(self._processes), MultiP2PQueue(self._processes)

    def repopulate_pool(self):
        # DOC INHERITED
        # for type hinting
        self._inqueue = self._inqueue  # type: MultiP2PQueue
        self._outqueue = self._outqueue  # type: MultiP2PQueue

        for i in range(self._processes - len(self._pool)):
            w = Process(
                target=self.worker,
                args=(
                    self._inqueue.get_sub_queue(i).get,
                    self._outqueue.get_sub_queue(i).put,
                    self._initializer,
                    self._initargs,
                    self._maxtasksperchild,
                ),
                ctx=self._ctx,
            )
            self._pool.append(w)
            w.name = w.name.replace("Process", "P2PPoolWorker")
            w.daemon = True
            w.start()
            default_logger.debug(f"Added worker {w.name}")

    def close(self):
        # we cannot rely on sentinels to shutdown worker processes
        # since there is no guarantee that each worker will get 1
        # "None" sentinel
        self.terminate()

    def __reduce__(self):
        raise NotImplementedError(
            "P2PPool objects cannot be passed between processes or pickled"
        )


class CtxPoolStorage:
    """
    This storage class is used by all :class:`.CtxPool` instances.
    However, since for each worker process, they have different
    memory spaces, ``storage`` is unique for all workers.

    ``storage`` is accessed on the client process side.
    """

    storage = None


class CtxPool(Pool):
    """
    Pool with context for each worker. your function must accept a ``ctx``
    object as your first non-keyword argument.

    If ``worker_contexts`` is not specified, then ``ctx`` will be ``None``.

    The length of ``worker_contexts`` must be the same as ``processes``
    """

    def __init__(
        self,
        processes: int,
        initializer=None,
        initargs=(),
        maxtasksperchild=None,
        worker_contexts=None,
        is_recursive=False,
        is_daemon=True,
        is_copy_tensor=True,
        share_method=None,
    ):

        if worker_contexts is not None:
            if len(worker_contexts) != processes:
                raise ValueError(
                    "Context number is not equal to the number of " "pool workers."
                )
        else:
            worker_contexts = [None] * processes

        super().__init__(
            processes=processes,
            initializer=self._init_with_context,
            initargs=(worker_contexts, initializer) + initargs,
            maxtasksperchild=maxtasksperchild,
            is_recursive=is_recursive,
            is_daemon=is_daemon,
            is_copy_tensor=is_copy_tensor,
            share_method=share_method,
        )
        self._caller = proxy_ctx_caller

    def repopulate_pool(self):
        """
        Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        # Get existing process ids:
        ids = {p.id for p in self._pool}
        need_ids = set(range(self._processes)) - ids

        for _, pid in zip(range(self._processes - len(self._pool)), need_ids):
            initargs = list(self._initargs)

            # Unpack context
            initargs[0] = initargs[0][pid]

            args = (
                self._inqueue.get,
                self._outqueue.put,
                self._initializer,
                initargs,
                self._maxtasksperchild,
            )

            if hasattr(self, "_wrap_exception"):
                args += (self._wrap_exception,)

            # changed worker -> clean_worker
            worker = Process(target=self.worker, args=args)
            worker.id = pid
            self._pool.append(worker)
            worker.name = worker.name.replace("Process", "CtxPoolWorker")
            worker.daemon = self._is_daemon
            worker.start()
            default_logger.debug("Added worker")

    @staticmethod
    def _init_with_context(context, init_func, *initargs):
        CtxPoolStorage.storage = context
        if init_func is not None:
            init_func(*initargs)

    def __reduce__(self):
        raise NotImplementedError(
            "CtxPool objects cannot be passed between processes or pickled"
        )


class ThreadPool(Pool):
    """
    A typical thread pool.
    """

    # Seems that manually adding gc
    # (when using torch.multiprocessing.pool.clean_worker as worker function)
    # will cause thread-pool to hang on this function on exit:
    # _wait_for_tstate_lock()

    def setup_queues(self):
        return queue.SimpleQueue(), queue.SimpleQueue()

    def pool_inqueue_put(self, obj: Any):
        return self._inqueue.put(obj)

    def pool_outqueue_get(self, timeout: float):
        try:
            return self._outqueue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError()

    def terminate_workers(self):
        """
        You can't and shouldn't terminate python threads.
        """
        pass

    def join_workers(self):
        for idx, worker in enumerate(self._pool):
            if worker.is_alive():
                # worker has not yet exited
                default_logger.debug(f"Cleaning up worker with id {worker.id}")
                worker.join()

    def maintain_pool(self):
        """
        Watch workers for exceptions and raise them and then terminate the pool,
        Clean up any retired workers reaching max task number, and
        start replacements for them.

        Override this method to implement your own pool.
        """
        for i in reversed(range(len(self._pool))):
            worker = self._pool[i]
            if worker.exception is not None:
                default_logger.critical(worker.exception, exc_info=True)

            if not worker.is_alive():
                # worker exited
                default_logger.debug(f"Cleaning up worker with id {worker.id}")
                worker.join()
                del self._pool[i]
        self.repopulate_pool()

    def repopulate_pool(self):
        """
        Bring the number of pool workers up to the specified number,
        it also creates new workers to replace old workers which have
        exited after executing ``maxtasksperchild``.

        Override this method to implement your own pool.
        """
        ids = {t.id for t in self._pool}
        need_ids = set(range(self._processes)) - ids

        for _, tid in zip(range(self._processes - len(self._pool)), need_ids):
            worker = Thread(
                target=self.worker,
                args=(
                    self._inqueue.get,
                    self._outqueue.put,
                    self._initializer,
                    self._initargs,
                    self._maxtasksperchild,
                ),
            )
            self._pool.append(worker)
            worker.daemon = True
            worker.id = tid
            worker.start()
            default_logger.debug(f"Added worker thread with id {tid}")

    def __reduce__(self):
        raise NotImplementedError(
            "ThreadPool objects cannot be passed between processes or pickled"
        )


class CtxThreadPool(ThreadPool):
    _context = threading.local()

    def __init__(
        self, processes: int, initializer=None, initargs=(), worker_contexts=None
    ):
        if worker_contexts is not None:
            if len(worker_contexts) != processes:
                raise ValueError(
                    "Context number is not equal to the number of " "pool workers."
                )
        else:
            worker_contexts = [None] * processes

        super().__init__(
            processes=processes,
            initializer=self._init_with_context,
            initargs=(worker_contexts, initializer) + initargs,
        )

    def apply(self, func, args=(), kwds=None):
        if kwds is None:
            kwds = {}
        return super().apply_async(self._wrap_func(func), args, kwds).get()

    def apply_async(self, func, args=(), kwds=None, callback=None, error_callback=None):
        if kwds is None:
            kwds = {}
        return super().apply_async(
            self._wrap_func(func), args, kwds, callback, error_callback
        )

    def map(self, func, iterable, chunksize=None):
        return super().map(self._wrap_func(func), iterable, chunksize)

    def map_async(
        self, func, iterable, chunksize=None, callback=None, error_callback=None
    ):
        return super().map_async(
            self._wrap_func(func), iterable, chunksize, callback, error_callback
        )

    def imap(self, func, iterable, chunksize=1):
        return super().imap(self._wrap_func(func), iterable, chunksize)

    def imap_unordered(self, func, iterable, chunksize=1):
        return super().imap_unordered(self._wrap_func(func), iterable, chunksize)

    def starmap(self, func, iterable, chunksize=None):
        return super().starmap(self._wrap_func(func), iterable, chunksize)

    def starmap_async(
        self, func, iterable, chunksize=None, callback=None, error_callback=None
    ):
        return super().starmap_async(
            self._wrap_func(func), iterable, chunksize, callback, error_callback
        )

    def repopulate_pool(self):
        """
        Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        # Get existing thread ids:
        ids = {t.id for t in self._pool}
        need_ids = set(range(self._processes)) - ids

        for _, tid in zip(range(self._processes - len(self._pool)), need_ids):
            initargs = list(self._initargs)

            # Unpack context
            initargs[0] = initargs[0][tid]

            args = (
                self._inqueue.get,
                self._outqueue.put,
                self._initializer,
                initargs,
                self._maxtasksperchild,
            )

            if hasattr(self, "_wrap_exception"):
                args += (self._wrap_exception,)

            # changed worker -> clean_worker
            worker = Thread(target=self.worker, args=args)
            worker.daemon = True
            worker.id = tid
            self._pool.append(worker)
            worker.start()
            default_logger.debug(f"Added worker thread with id {tid}")

    @classmethod
    def _wrap_func(cls, func):
        def call(*args, **kwargs):
            ctx = cls._context.storage
            return func(ctx, *args, **kwargs)

        return call

    @staticmethod
    def _init_with_context(context, init_func, *initargs):
        CtxThreadPool._context.storage = context
        if init_func is not None:
            init_func(*initargs)

    def __reduce__(self):
        raise NotImplementedError(
            "CtxThreadPool objects cannot be passed between processes or pickled"
        )
