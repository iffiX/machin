from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Lock
from copy import deepcopy
from ..distributed import RpcGroup


class OrderedServerBase(ABC):  # pragma: no cover
    """
    Descendent classes of OrderedServer does not have to guarantee strong
    consistency, that is, even if :meth:`.OrderedServerBase.push_service``
    has returned True, there are possibilities that these acknowledged
    push are discarded.
    """
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


class OrderedServerSimple(OrderedServerBase):
    """
    A simple key-value server, with strict ordered update
    """
    def __init__(self,
                 server_name: str,
                 server_runner: str,
                 group: RpcGroup,
                 version_depth: int = 1,
                 **__):
        """
        Args:
            server_name: Name of this server, used to registered
                the server as a paired class of ``group``.
            server_role: Name of the process serving the ordered server.
            group: Rpc group where server locates.
            version_depth: Storage depth of old versions of the same
                key. If ``depth = 1``, then only the newest version
                of the key will be saved.
        """
        assert group.is_member(server_runner)
        assert version_depth > 0 and isinstance(version_depth, int)

        self.server_name = server_name
        self.server_runner = server_runner
        self.group = group
        self.storage = {}
        self.lock = Lock()
        self.log_depth = version_depth
        self.group.rpc_pair(server_name, self)

    def push(self, key, value, version, prev_version):
        # DOC INHERITED
        return self.group.rpc_paired_class_sync(
            self.server_runner, self.server_name, self._push_service,
            args=(key, value, version, prev_version)
        )

    def pull(self, key, version=None):
        # DOC INHERITED
        return self.group.rpc_paired_class_sync(
            self.server_runner, self.server_name, self._pull_service,
            args=(key, version)
        )

    def _push_service(self, key, value, version, prev_version):
        success = False
        with self.lock:
            if key in self.storage:
                ref = self.storage[key]
                # Check previous version consistency.
                if next(reversed(ref)) == prev_version:
                    ref[version] = value
                    success = True
                if len(ref) > self.log_depth + 1:
                    ref.popitem(last=False)
            else:
                # Create a new key.
                ref = self.storage[key] = OrderedDict()
                ref[version] = value
                success = True
        return success

    def _pull_service(self, key, version=None):
        result = None
        with self.lock:
            if key in self.storage:
                ref = self.storage[key]
                # Try to find the target version.
                if version is not None and version in ref:
                    result = (deepcopy(ref[version]), version)
                # Find the newest version.
                elif version is None:
                    version = next(reversed(ref))
                    result = (deepcopy(ref[version]), version)
        return result
