import multiprocessing.pool as pool
from torch.multiprocessing.pool import clean_worker

import dill
from .queue import SimpleQueue


def proxy_caller(input_):
    """
    Call a serialized function and return results.
    """
    func_str, args, kwargs = input_
    func = dill.loads(func_str)
    return func(*args, **kwargs)


def proxy_dumper(recurse, func, args_list, kwargs_list=None):
    """
    Serialize a function so it can be called.

    Returns:
        List[function string, arguments...]
    """
    # recurse will enable context variable saving
    dump = dill.dumps(func, protocol=dill.HIGHEST_PROTOCOL, recurse=recurse)
    if kwargs_list is not None:
        for args, kwargs in zip(args_list, kwargs_list):
            yield [dump, args, kwargs]
    else:
        for args in args_list:
            yield [dump, args, {}]


class Pool(pool.Pool):
    """
    Enhanced multiprocessing pool for pytorch, provides:
     1. Support for lambdas and local functions.
     2. Ability to select the tensor serialize scheme.
     3. Ability to get the number of workers.
    """

    # Multiprocessing pool is badly written.
    # python IDEs will complain a lot.

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None,
                 is_global=True, is_daemon=True, is_copy_tensors=True):
        self.is_global = is_global
        self.is_daemon = is_daemon
        self.is_copy_tensors = is_copy_tensors
        super(Pool, self).__init__(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild,
            context=context
        )

    def _setup_queues(self):
        self._inqueue = SimpleQueue(ctx=self._ctx,
                                    copy_tensor=self.is_copy_tensors)
        self._outqueue = SimpleQueue(ctx=self._ctx,
                                     copy_tensor=self.is_copy_tensors)
        self._quick_put = self._inqueue.quick_put
        self._quick_get = self._outqueue.quick_get

    def apply(self, func, args=(), kwds=None):
        # DOC INHERITED
        if kwds is None:
            kwds = {}
        return pool.Pool.apply(self, proxy_caller,
                               [dill.dumps(func, recurse=self.is_global),
                                args, kwds])

    def apply_async(self, func, args=(), kwds=None, callback=None,
                    error_callback=None):
        # DOC INHERITED
        if kwds is None:
            kwds = {}
        return pool.Pool.apply_async(self, proxy_caller,
                                     [dill.dumps(func, recurse=self.is_global),
                                      args, kwds])

    def map(self, func, iterable, chunksize=None):
        # DOC INHERITED
        return pool.Pool.map(self, proxy_caller,
                             proxy_dumper(
                                 self.is_global,
                                 func,
                                 [(arg,) for arg in iterable]
                             ),
                             chunksize)

    def map_async(self, func, iterable, chunksize=None, callback=None,
                  error_callback=None):
        # DOC INHERITED
        return pool.Pool.map_async(self, proxy_caller,
                                   proxy_dumper(
                                       self.is_global,
                                       func,
                                       [(arg,) for arg in iterable]
                                   ),
                                   chunksize, callback, error_callback)

    def imap(self, func, iterable, chunksize=1):
        # DOC INHERITED
        return pool.Pool.imap(self, proxy_caller,
                              proxy_dumper(
                                  self.is_global,
                                  func,
                                  [(arg,) for arg in iterable]
                              ),
                              chunksize)

    def imap_unordered(self, func, iterable, chunksize=1):
        # DOC INHERITED
        return pool.Pool.imap_unordered(self, proxy_caller,
                                        proxy_dumper(
                                            self.is_global,
                                            func,
                                            [(arg,) for arg in iterable]
                                        ),
                                        chunksize)

    def starmap(self, func, iterable, chunksize=None):
        # DOC INHERITED
        return pool.Pool.starmap(self, proxy_caller,
                                 proxy_dumper(
                                     self.is_global,
                                     func,
                                     iterable
                                 ),
                                 chunksize)

    def starmap_async(self, func, iterable, chunksize=None, callback=None,
                      error_callback=None):
        # DOC INHERITED
        return pool.Pool.starmap_async(self, proxy_caller,
                                       proxy_dumper(
                                           self.is_global,
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
            worker.daemon = self.is_daemon
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


class ThreadPool(pool.ThreadPool):
    """
    A typical thread pool, provides:
    1. Ability to get the number of workers.
    """
    # Multiprocessing pool is badly written.
    # python IDEs will complain a lot.

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
            worker.start()
            pool.util.debug('added worker')

    def size(self):
        """
        Returns:
            The number of workers in pool.
        """
        return len(self._pool)

    def __reduce__(self):
        raise RuntimeError("Thread pool is not reducible.")
