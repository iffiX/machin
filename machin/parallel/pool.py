from typing import Any, List
import threading
import multiprocessing.pool as pool
from torch.multiprocessing.pool import clean_worker

import dill
from .queue import SimpleQueue


def proxy_caller(*input_):
    """
    Call a serialized function and return results.
    """
    if len(input_) == 1:
        func_str, args, kwargs = input_[0]
    else:
        func_str, args, kwargs = input_
    func = dill.loads(func_str)
    return func(*args, **kwargs)


def proxy_ctx_caller(*input_):
    """
    Call a serialized function with worker context and return results.
    """
    if len(input_) == 1:
        func_str, args, kwargs = input_[0]
    else:
        func_str, args, kwargs = input_
    func = dill.loads(func_str)
    return func(CtxPoolStorage.storage, *args, **kwargs)


def proxy_dumper(recurse, func, args_list):
    """
    Serialize a function so it can be called.

    Returns:
        List[function string, arguments...]
    """
    # recurse will enable context variable saving
    dump = dill.dumps(func, protocol=dill.HIGHEST_PROTOCOL, recurse=recurse)
    for args in args_list:
        yield [dump, args, {}]


class Pool(pool.Pool):
    """
    Enhanced multiprocessing pool for pytorch, provides:
     1. Support for lambdas and local functions.
     2. Ability to select the tensor serialize scheme.
    """

    # Multiprocessing pool is badly written.
    # python IDEs will complain a lot.

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None,
                 is_global=False, is_daemon=True, is_copy_tensors=False):
        self._is_global = is_global
        self._is_daemon = is_daemon
        self._is_copy_tensors = is_copy_tensors
        self._caller = proxy_caller
        super(Pool, self).__init__(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild,
            context=context
        )

    def _setup_queues(self):
        self._inqueue = SimpleQueue(ctx=self._ctx,
                                    copy_tensor=self._is_copy_tensors)
        self._outqueue = SimpleQueue(ctx=self._ctx,
                                     copy_tensor=self._is_copy_tensors)
        self._quick_put = self._inqueue.quick_put
        self._quick_get = self._outqueue.quick_get

    def apply(self, func, args=(), kwds=None):
        # DOC INHERITED
        if kwds is None:
            kwds = {}
        return pool.Pool.apply(self, self._caller,
                               [(dill.dumps(
                                   func, recurse=self._is_global),
                                args, kwds)])

    def apply_async(self, func, args=(), kwds=None, callback=None,
                    error_callback=None):
        # DOC INHERITED
        if kwds is None:
            kwds = {}
        return pool.Pool.apply_async(self, self._caller,
                                     [(dill.dumps(
                                         func, recurse=self._is_global),
                                      args, kwds)])

    def map(self, func, iterable, chunksize=None):
        # DOC INHERITED
        return pool.Pool.map(self, self._caller,
                             proxy_dumper(
                                 self._is_global,
                                 func,
                                 [(arg,) for arg in iterable]
                             ),
                             chunksize)

    def map_async(self, func, iterable, chunksize=None, callback=None,
                  error_callback=None):
        # DOC INHERITED
        return pool.Pool.map_async(self, self._caller,
                                   proxy_dumper(
                                       self._is_global,
                                       func,
                                       [(arg,) for arg in iterable]
                                   ),
                                   chunksize, callback, error_callback)

    def imap(self, func, iterable, chunksize=1):
        # DOC INHERITED
        return pool.Pool.imap(self, self._caller,
                              proxy_dumper(
                                  self._is_global,
                                  func,
                                  [(arg,) for arg in iterable]
                              ),
                              chunksize)

    def imap_unordered(self, func, iterable, chunksize=1):
        # DOC INHERITED
        return pool.Pool.imap_unordered(self, self._caller,
                                        proxy_dumper(
                                            self._is_global,
                                            func,
                                            [(arg,) for arg in iterable]
                                        ),
                                        chunksize)

    def starmap(self, func, iterable, chunksize=None):
        # DOC INHERITED
        return pool.Pool.starmap(self, self._caller,
                                 proxy_dumper(
                                     self._is_global,
                                     func,
                                     iterable
                                 ),
                                 chunksize)

    def starmap_async(self, func, iterable, chunksize=None, callback=None,
                      error_callback=None):
        # DOC INHERITED
        return pool.Pool.starmap_async(self, self._caller,
                                       proxy_dumper(
                                           self._is_global,
                                           func,
                                           iterable
                                       ),
                                       chunksize, callback, error_callback)

    def _repopulate_pool(self):
        """
        Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for _ in range(self._processes - len(self._pool)):
            # changed worker -> clean_worker
            args = (self._inqueue, self._outqueue,
                    self._initializer,
                    self._initargs, self._maxtasksperchild)
            if hasattr(self, '_wrap_exception'):
                args += (self._wrap_exception,)
            worker = self.Process(target=clean_worker, args=args)
            self._pool.append(worker)
            worker.name = worker.name.replace('Process', 'PoolWorker')
            worker.daemon = self._is_daemon
            worker.start()
            pool.util.debug('added worker')

    def size(self):
        """
        Returns:
            The number of workers in pool.
        """
        return len(self._pool)

    def __reduce__(self):
        raise RuntimeError("Process pool is not reducible.")


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
    def __init__(self, processes: int, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None, worker_contexts=None,
                 is_global=False, is_daemon=True, is_copy_tensors=False):

        if worker_contexts is not None:
            if len(worker_contexts) != processes:
                raise ValueError("Context number is not equal to the number of "
                                 "pool workers.")
        else:
            worker_contexts = [None] * processes

        super(CtxPool, self).__init__(
            processes=processes,
            initializer=self._init_with_context,
            initargs=(worker_contexts, initializer) + initargs,
            maxtasksperchild=maxtasksperchild,
            context=context,
            is_global=is_global,
            is_daemon=is_daemon,
            is_copy_tensors=is_copy_tensors
        )
        self._caller = proxy_ctx_caller

    def _repopulate_pool(self):
        """
        Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        # Get existing process ids:
        ids = {p.id for p in self._pool}
        need_ids = set(range(self._processes)) - ids

        for _, id in zip(range(self._processes - len(self._pool)), need_ids):
            initargs = list(self._initargs)

            # Unpack context
            initargs[0] = initargs[0][id]

            args = (self._inqueue, self._outqueue,
                    self._initializer,
                    initargs, self._maxtasksperchild)

            if hasattr(self, '_wrap_exception'):
                args += (self._wrap_exception,)

            # changed worker -> clean_worker
            worker = self.Process(target=clean_worker, args=args)
            worker.id = id
            self._pool.append(worker)
            worker.name = worker.name.replace('Process', 'CtxPoolWorker')
            worker.daemon = self._is_daemon
            worker.start()
            pool.util.debug('added worker')

    @staticmethod
    def _init_with_context(context, init_func, *initargs):
        CtxPoolStorage.storage = context
        if init_func is not None:
            init_func(*initargs)


class ThreadPool(pool.ThreadPool):
    """
    A typical thread pool.
    """
    # Multiprocessing pool is badly written.
    # python IDEs will complain a lot.

    # Seems that manually adding gc
    # (when using torch.multiprocessing.pool.clean_worker as worker function)
    # will cause thread-pool to hang on this function on exit:
    # _wait_for_tstate_lock()

    # so _repopulate_pool is not overloaded

    def size(self):
        """
        Returns:
            The number of workers in pool.
        """
        return len(self._pool)

    def __reduce__(self):
        raise RuntimeError("Thread pool is not reducible.")


class CtxThreadPool(ThreadPool):
    _context = threading.local()

    def __init__(self, processes: int, initializer=None, initargs=(),
                 worker_contexts=None):
        if worker_contexts is not None:
            if len(worker_contexts) != processes:
                raise ValueError("Context number is not equal to the number of "
                                 "pool workers.")
        else:
            worker_contexts = [None] * processes

        super(CtxThreadPool, self).__init__(
            processes=processes,
            initializer=self._init_with_context,
            initargs=(worker_contexts, initializer) + initargs
        )

    def apply(self, func, args=(), kwds=None):
        if kwds is None:
            kwds = {}
        return super(CtxThreadPool, self).apply_async(
            self._wrap_func(func), args, kwds
        ).get()

    def apply_async(self, func, args=(), kwds=None, callback=None,
                    error_callback=None):
        if kwds is None:
            kwds = {}
        return super(CtxThreadPool, self).apply_async(
            self._wrap_func(func), args, kwds, callback, error_callback
        )

    def map(self, func, iterable, chunksize=None):
        return super(CtxThreadPool, self).map(
            self._wrap_func(func), iterable, chunksize
        )

    def map_async(self, func, iterable, chunksize=None, callback=None,
                  error_callback=None):
        return super(CtxThreadPool, self).map_async(
            self._wrap_func(func), iterable, chunksize,
            callback, error_callback
        )

    def imap(self, func, iterable, chunksize=1):
        return super(CtxThreadPool, self).imap(
            self._wrap_func(func), iterable, chunksize
        )

    def imap_unordered(self, func, iterable, chunksize=1):
        return super(CtxThreadPool, self).imap_unordered(
            self._wrap_func(func), iterable, chunksize
        )

    def starmap(self, func, iterable, chunksize=None):
        return super(CtxThreadPool, self).starmap(
            self._wrap_func(func), iterable, chunksize
        )

    def starmap_async(self, func, iterable, chunksize=None, callback=None,
                      error_callback=None):
        return super(CtxThreadPool, self).starmap_async(
            self._wrap_func(func), iterable, chunksize,
            callback, error_callback
        )

    def _repopulate_pool(self):
        """
        Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        # Get existing process ids:
        ids = {p.id for p in self._pool}
        need_ids = set(range(self._processes)) - ids

        for _, id in zip(range(self._processes - len(self._pool)), need_ids):
            initargs = list(self._initargs)

            # Unpack context
            initargs[0] = initargs[0][id]

            args = (self._inqueue, self._outqueue,
                    self._initializer,
                    initargs, self._maxtasksperchild)

            if hasattr(self, '_wrap_exception'):
                args += (self._wrap_exception,)

            # changed worker -> clean_worker
            worker = self.Process(target=pool.worker, args=args)
            worker.id = id
            self._pool.append(worker)
            worker.name = worker.name.replace('Process', 'CtxThreadPoolWorker')
            worker.start()
            pool.util.debug('added worker')

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
