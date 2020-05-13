import io
import sys
import dill
from torch.multiprocessing.queue import ForkingPickler
from multiprocessing import context, connection


class ConnectionWrapper(object):
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler to
    serialize objects"""

    """
        Note: dumping is not recursive
    """

    def __init__(self, conn):
        self.conn = conn
        self.copy_tensor = False

    def set_copy(self, copy_tensor):
        self.copy_tensor = copy_tensor

    def send(self, obj):
        if not self.copy_tensor:
            buf = io.BytesIO()
            ForkingPickler(buf, dill.HIGHEST_PROTOCOL).dump(obj)
            self.send_bytes(buf.getvalue())
        else:
            self.send_bytes(dill.dumps(obj))

    def recv(self):
        buf = self.recv_bytes()
        return dill.loads(buf)

    def send_bytes(self, bytes):
        self.conn.send_bytes(bytes)

    def recv_bytes(self, timeout=360):
        if self.conn.poll(timeout=timeout):
            return self.conn.recv_bytes()
        else:
            raise RuntimeError("Timeout")

    def __getattr__(self, name):
        if 'conn' in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, 'conn'))


class SimpleQueue(object):
    def __init__(self, *, ctx):
        self._reader, self._writer = connection.Pipe(duplex=False)
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._rlock = ctx.Lock()
        self._poll = self._reader.poll
        if sys.platform == 'win32':
            self._wlock = None
        else:
            self._wlock = ctx.Lock()

    def set_copy(self, copy_tensor):
        self._reader.set_copy(copy_tensor)
        self._writer.set_copy(copy_tensor)

    def empty(self):
        return not self._poll()

    def __getstate__(self):
        context.assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock)

    def __setstate__(self, state):
        (self._reader, self._writer, self._rlock, self._wlock) = state

    def get(self):
        with self._rlock:
            res = self._reader.recv_bytes()
        # unserialize the data after having released the lock
        return dill.loads(res)

    def put(self, obj):
        # serialize the data before acquiring the lock
        obj = dill.dumps(obj)
        if self._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self._writer.send_bytes(obj)
        else:
            with self._wlock:
                self._writer.send_bytes(obj)
