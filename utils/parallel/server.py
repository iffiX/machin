from collections import OrderedDict
from threading import Lock
from .distributed import Group


class SimpleOrderedServer:
    def __init__(self, group: Group, log_depth=1, *_, **__):
        """
        Currently, only one process is used, and provides no functionality
        for serialization, snapshot & log compaction, redundancy...
        """
        self.group = group
        self.group.rpc_register_paired(self.__class__, self)
        self.storage = {}
        self.lock = Lock()
        self.log_depth = log_depth

        assert log_depth > 0 and isinstance(log_depth, int)

    def push(self, key, value, version, prev_version):
        to = self.group.get_peer_ranks()[0]
        return self.group.rpc_paired_class_sync(to, self._reply_push,
                                                args=(key, value, version, prev_version))

    def pull(self, key, version=None):
        to = self.group.get_peer_ranks()[0]
        return self.group.rpc_paired_class_sync(to, self._reply_pull,
                                                args=(key, version))

    def latest(self, key):
        to = self.group.get_peer_ranks()[0]
        return self.group.rpc_paired_class_sync(to, self._reply_pull,
                                                args=(key, hash))

    def _reply_push(self, key, value, version, prev_version):
        success = False
        self.lock.acquire()
        if key in self.storage:
            ref = self.storage[key]
            if next(reversed(ref)) == prev_version:
                ref[version] = value
                success = True
            if len(ref) > self.log_depth + 1:
                ref.pop(0)
        else:
            ref = self.storage[key] = OrderedDict()
            ref[version] = value
            success = True
        self.lock.release()
        return success

    def _reply_pull(self, key, version):
        result = None
        self.lock.acquire()
        if key in self.storage:
            ref = self.storage[key]
            if version is not None and version in ref:
                result = (version, ref[version])
            elif version is None:
                result = (next(reversed(ref)), ref[-1])
        self.lock.release()
        return result

    def _reply_latest(self, key):
        result = None
        self.lock.acquire()
        if key in self.storage:
            ref = self.storage[key]
            result = next(reversed(ref))
        self.lock.release()
        return result
