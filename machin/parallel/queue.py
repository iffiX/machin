from time import time
from typing import Any
from multiprocessing import context, connection, get_context
import sys

from .pickle import dumps, loads


class ConnectionWrapper:  # pragma: no cover
    """
    This simple wrapper provides timeout function for sending
    bytes on ``Connection``.
    """

    def __init__(self, conn):
        self.conn = conn

    def send_bytes(self, bytes_):
        """
        Send bytes over the connection pipe.
        """
        self.conn.send_bytes(bytes_)

    def recv_bytes(self, timeout=None):
        """
        Receive bytes from the connection pipe.

        Raises:
            TimeoutError if timeout.
        """
        if self.conn.poll(timeout=timeout):
            return self.conn.recv_bytes()
        else:
            raise TimeoutError("Timeout")

    def __getattr__(self, name):
        if "conn" in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'conn'")


class SimpleQueue:  # pragma: no cover
    """
    A simple single direction queue for inter-process communications.
    There could be multiple receivers and multiple senders on each side.
    """

    def __init__(self, *, ctx=None, copy_tensor=False):
        """
        Args:
            ctx: Multiprocessing context, you can get this using ``get_context``
            copy_tensor: Set the queue to send a fully serialized tensor
                if ``True``, and only a stub of reference if ``False``.

        See Also:
            :func:`.dump_tensor`
        """
        if ctx is None:
            # get default context
            ctx = get_context()
        self._reader, self._writer = connection.Pipe(duplex=False)
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        # _rlock will be used by _help_stuff_finish() of multiprocessing.Pool
        self._rlock = ctx.Lock()
        self._copy_tensor = copy_tensor
        if sys.platform == "win32":
            self._wlock = None
        else:
            self._wlock = ctx.Lock()

    def empty(self):
        """
        Returns:
            Whether the queue is empty or not.
        """
        return not self._reader.poll()

    def close(self):
        self._reader.close()
        self._writer.close()

    def __getstate__(self):
        context.assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock, self._copy_tensor)

    def __setstate__(self, state):
        (
            self._reader,
            self._writer,
            self._rlock,
            self._wlock,
            self._copy_tensor,
        ) = state

    def get(self, timeout=None):
        """
        Get an object from the queue. This api is required by
        ``multiprocessing.pool`` to perform inter-process
        communication.

        Note:
            This api is used by sub-processes in pool to get tasks
            and work.

        Returns:
            Any object.
        """
        with self._rlock:
            res = self._reader.recv_bytes(timeout)
        # deserialize the data after having released the lock
        return loads(res)

    def put(self, obj: Any):
        """
        Put an object into the queue. This api is required by
        ``multiprocessing.pool`` to perform inter-process
        communication.

        Note:
            This api is used by sub-processes in pool to put results
            and respond.

        Args:
            obj: Any object.
        """
        # serialize the data before acquiring the lock
        obj = dumps(obj, copy_tensor=self._copy_tensor)
        if self._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self._writer.send_bytes(obj)
        else:
            with self._wlock:
                self._writer.send_bytes(obj)

    def quick_get(self, timeout=None):
        """
        Get an object from the queue.

        Note:
            this api is used by the result manager (``Pool._result_handler``)
            thread to get results from the queue, since it is single threaded,
            there is no need to use locks, and therefore quicker.

        Returns:
            Any object.
        """
        res = self._reader.recv_bytes(timeout)
        return loads(res)

    def quick_put(self, obj: Any):
        """
        Put an object into the queue.

        Note: this api is used by the pool manager (``Pool._task_handler``)
            thread to put tasks into the queue, since it is single threaded,
            there is no need to use locks, and therefore quicker.

        Args:
            obj: Any object.
        """
        obj = dumps(obj, copy_tensor=self._copy_tensor)
        self._writer.send_bytes(obj)

    def __del__(self):
        self.close()


class SimpleP2PQueue:  # pragma: no cover
    """
    A simple single direction queue for inter-process P2P communications.
    Each end only have one process.
    """

    def __init__(self, *, copy_tensor=False):
        """
        Args:
            copy_tensor: Set the queue to send a fully serialized tensor
                if ``True``, and only a stub of reference if ``False``.

        See Also:
            :func:`.dump_tensor`
        """
        self._reader, self._writer = connection.Pipe(duplex=False)
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._copy_tensor = copy_tensor

    def empty(self):
        """
        Returns:
            Used by receiver, Whether the queue is empty or not.
        """
        return not self._reader.poll()

    def close(self):
        self._reader.close()
        self._writer.close()

    def __getstate__(self):
        context.assert_spawning(self)
        return self._reader, self._writer, self._copy_tensor

    def __setstate__(self, state):
        (self._reader, self._writer, self._copy_tensor) = state

    def get(self, timeout=None):
        """
        Get an object from the queue. This api is required by
        ``multiprocessing.pool`` to perform inter-process
        communication.

        Note:
            This api is used by sub-processes in pool to get tasks
            and work.

        Returns:
            Any object.
        """
        res = self._reader.recv_bytes(timeout)
        # deserialize the data after having released the lock
        return loads(res)

    def put(self, obj: Any):
        """
        Put an object into the queue. This api is required by
        ``multiprocessing.pool`` to perform inter-process
        communication.

        Note:
            This api is used by sub-processes in pool to put results
            and respond.

        Args:
            obj: Any object.
        """
        # serialize the data before acquiring the lock
        obj = dumps(obj, copy_tensor=self._copy_tensor)
        self._writer.send_bytes(obj)

    def __del__(self):
        self.close()


class MultiP2PQueue:
    """
    P2P queue which connects pool result manager and worker processes directly,
    with no lock.
    """

    def __init__(self, queue_num, *, copy_tensor=False):
        self.counter = 0
        self.queues = [
            SimpleP2PQueue(copy_tensor=copy_tensor) for _ in range(queue_num)
        ]

    def put(self, obj: Any):
        # randomly choose a worker's queue
        queue = self.queues[self.counter]
        self.counter = (self.counter + 1) % len(self.queues)
        queue.put(obj)

    def get(self, timeout=None):
        begin = time()
        while True:
            if timeout is not None and time() - begin > timeout:
                raise TimeoutError("Timeout")
            for queue in self.queues:
                try:
                    obj = queue.get(timeout=1e-3)
                    return obj
                except TimeoutError:
                    continue

    def get_sub_queue(self, index):
        return self.queues[index]

    def __del__(self):
        for q in self.queues:
            q.close()


__all__ = ["SimpleQueue"]
