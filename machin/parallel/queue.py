from typing import Any
from multiprocessing import context, connection
import sys
import dill
import torch as t

from .pickle import dump_tensor


class ConnectionWrapper(object):
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
        if 'conn' in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, 'conn'))


class SimpleQueue(object):
    """
    A simple queue for inter-process communications.
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

        self._reader, self._writer = connection.Pipe(duplex=False)
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._r_lock = ctx.Lock()
        self._poll = self._reader.poll
        self._copy_tensor = copy_tensor
        if sys.platform == 'win32':
            self._w_lock = None
        else:
            self._w_lock = ctx.Lock()

    def empty(self):
        """
        Returns:
            Whether the queue is empty or not.
        """
        return not self._poll()

    def __getstate__(self):
        context.assert_spawning(self)
        return self._reader, self._writer, self._r_lock, self._w_lock

    def __setstate__(self, state):
        (self._reader, self._writer, self._r_lock, self._w_lock) = state

    def get(self):
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
        with self._r_lock:
            res = self._reader.recv_bytes()
        # deserialize the data after having released the lock
        return dill.loads(res)

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
        if t.is_tensor(obj):
            obj = dump_tensor(obj, self._copy_tensor)
        else:
            obj = dill.dumps(obj)
        if self._w_lock is None:
            # writes to a message oriented win32 pipe are atomic
            self._writer.send_bytes(obj)
        else:
            with self._w_lock:
                self._writer.send_bytes(obj)

    def quick_get(self):
        """
        Get an object from the queue.

        Note:
            this api is used by the result manager (``Pool._result_handler``)
            thread to get results from the queue, since it is single threaded,
            there is no need to use locks, and therefore quicker.

        Returns:
            Any object.
        """
        res = self._reader.recv_bytes()
        return dill.loads(res)

    def quick_put(self, obj: Any):
        """
        Put an object into the queue.

        Note: this api is used by the pool manager (``Pool._task_handler``)
            thread to put tasks into the queue, since it is single threaded,
            there is no need to use locks, and therefore quicker.

        Args:
            obj: Any object.
        """
        if t.is_tensor(obj):
            obj = dump_tensor(obj, self._copy_tensor)
        else:
            obj = dill.dumps(obj)
        self._writer.send_bytes(obj)


__all__ = ["SimpleQueue"]
