from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Lock
from copy import deepcopy
from ..distributed import RpcGroup, debug_with_process


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
    def __init__(self, server_name: str, group: RpcGroup):
        self._push_service = server_name + "/_push_service"
        self._pull_service = server_name + "/_pull_service"
        self.group = group

    def push(self, key, value, version, prev_version):
        # DOC INHERITED
        debug_with_process(
            f"calling push service {self._push_service} "
            f"on group [{self.group.group_name}]"
        )
        return self.group.registered_sync(
            self._push_service, args=(key, value, version, prev_version)
        )

    def pull(self, key, version=None):
        # DOC INHERITED
        debug_with_process(
            f"calling pull service {self._push_service} "
            f"on group [{self.group.group_name}]"
        )
        return self.group.registered_sync(self._pull_service, args=(key, version))


class OrderedServerSimpleImpl:
    """
    A simple key-value server, with strict ordered update
    """

    def __init__(self, server_name: str, group: RpcGroup, version_depth: int = 1, **__):
        """
        This init function must be only invoked on the runner process,
        and the runner process must be a member process of ``group``.

        Args:
            server_name: Name of this server, used to registered
                the server as a paired class of ``group``.
            group: Rpc group where server locates.
            server_runner: Name of the process serving the ordered server.
                By default is the first member of the group.
            version_depth: Storage depth of old versions of the same
                key. If ``depth = 1``, then only the newest version
                of the key will be saved.
        """
        assert group.is_member()
        assert version_depth > 0 and isinstance(version_depth, int)

        self.server_name = server_name
        self.group = group
        self.storage = {}
        self.lock = Lock()
        self.version_depth = version_depth
        # pair an accessor to group
        self.group.pair(server_name, OrderedServerSimple(self.server_name, self.group))
        self.group.register(server_name + "/_push_service", self._push_service)
        self.group.register(server_name + "/_pull_service", self._pull_service)

    def _push_service(self, key, value, version, prev_version):
        success = False
        with self.lock:
            if key in self.storage:
                ref = self.storage[key]
                # Check previous version consistency.
                if next(reversed(ref)) == prev_version:
                    ref[version] = value
                    success = True
                if len(ref) > self.version_depth + 1:
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
