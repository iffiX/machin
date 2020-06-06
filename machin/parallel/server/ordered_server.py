from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Lock
from random import choice
from machin.parallel.distributed import RpcGroup


class OrderedServerBase(ABC):
    @abstractmethod
    def push(self, key, value, version, prev_version):
        """
        Push a new ``version`` of ``value`` in ``key`` to the ordered server.

        Note:
            If ``version = prev_version`` then there is no order guarantee. But
            you may exploit this feature.

        Args:
            key: Key.
            value: value.
            version: New version.
            prev_version: Previous version.

        Returns:
            ``True`` if success, and ``False`` if not.
        """
        pass

    @abstractmethod
    def pull(self, key, version=None):
        """
        Pull a value with the specified ``version`` in ``key``.

        Args:
            key: Key.
            version: Target version, if ``None``, then the newest version
                of value of key will be pulled.

        Returns:
            ``None`` if version is not found, auto-deleted, or key is not found,
            otherwise returns value with the specified ``version``
            in ``key``, and then ``version``
        """
        pass


class SimpleOrderedServer(OrderedServerBase):
    """
    A simple key-value server, with strict ordered update and automatic
    replication.

    Note:
        This simple implementation is based on a master-slave architecture,
        and only provides availability, but provides no guarantee on
        consistency, if master fails, the whole server group
        will fail.
    """
    def __init__(self,
                 group: RpcGroup,
                 master: int = 0,
                 version_depth: int = 1,
                 **__):
        """
        Args:
            group: The rpc group of this server.
            master: The relative rpc rank of the master server. Must
                be less than group size.
            version_depth: Storage depth of old versions of the same
                key. If ``depth = 1``, then only the newest version
                of the key will be saved.
        """
        self.group = group
        self.group.rpc_register_paired(self.__class__, self)
        self.storage = {}
        self.lock = Lock()
        self.log_depth = version_depth
        self.master = self.group.get_group_members()[master]
        self.slaves = [member
                       for member in self.group.get_group_members()
                       if member != self.master]

        assert version_depth > 0 and isinstance(version_depth, int)

    def push(self, key, value, version, prev_version):
        # DOC INHERITED
        return self.group.rpc_paired_class_sync(self.master,
                                                self._reply_push,
                                                self.__class__,
                                                args=(key, value, version,
                                                      prev_version))

    def pull(self, key, version=None):
        # DOC INHERITED
        to = choice(self.group.get_group_members())
        return self.group.rpc_paired_class_sync(to,
                                                self._reply_pull,
                                                self.__class__,
                                                args=(key, version))[1]

    def _reply_push(self, key, value, version, prev_version):
        success = False
        self.lock.acquire()
        if key in self.storage:
            ref = self.storage[key]
            # Check previous version consistency.
            if next(reversed(ref)) == prev_version:
                ref[version] = value
                success = True
            if len(ref) > self.log_depth + 1:
                ref.pop(0)
        else:
            # Create a new key.
            ref = self.storage[key] = OrderedDict()
            ref[version] = value
            success = True

        future = []
        for slave in self.slaves:
            # push updates to slaves
            future.append(self.group.rpc_paired_class_sync(
                slave, self._master_sync, self.__class__,
                args=(key, value, version)
            ))

        self.lock.release()
        return success

    def _reply_pull(self, key, version):
        result = None
        self.lock.acquire()
        if key in self.storage:
            ref = self.storage[key]
            # Try to find the target version.
            if version is not None and version in ref:
                result = (ref[version], version)
            # Find the newest version.
            elif version is None:
                result = (ref[-1], next(reversed(ref)))
        self.lock.release()
        return result

    def _master_sync(self, key, value, version):
        # faster _reply_push, for master-slave sync
        if key in self.storage:
            ref = self.storage[key]
            ref[version] = value
            if len(ref) > self.log_depth + 1:
                ref.pop(0)
        else:
            ref = self.storage[key] = OrderedDict()
            ref[version] = value
