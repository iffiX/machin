from threading import Lock, Event
from datetime import timedelta
from typing import Union, List, Any, Callable
from inspect import getframeinfo, stack
from time import sleep
from torch.distributed import rpc
from machin.parallel.exception import ExceptionWithTraceback
from machin.utils.logging import default_logger
from logging import DEBUG

import enum
import torch as t
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

WORLD = None  # type: Union[None, World]


class LUTType(enum.Enum):
    VALUE = 0
    SERVICE = 1


def debug_with_process(message):
    if default_logger.level != DEBUG:
        return
    caller = getframeinfo(stack()[1][0])
    default_logger.debug(
        f"Process [{get_cur_rank()}]: "
        f"<{caller.filename},L{caller.lineno}> "
        f"{message}"
    )


def _copy_doc(from_func):
    """
    Used by collective group to copy documents from torch.
    """
    import io
    import sys

    def _decorator(func):
        if "sphinx" in sys.modules:  # pragma: no cover
            src_doc = from_func.__doc__
            lines = io.StringIO(src_doc)
            # remove the group line
            src_doc = "".join([line for line in lines if "group" not in line])
            func.__doc__ = src_doc
        return func

    return _decorator


def _rpc_set_lut_entry(group_name, key, proc_name, lut_type):  # pragma: no cover
    table = WORLD.value_lut if lut_type == LUTType.VALUE else WORLD.service_lut
    with WORLD.lut_lock:
        if (group_name, key) in table:
            return False
        else:
            table[(group_name, key)] = proc_name
            return True


def _rpc_unset_lut_entry(group_name, key, proc_name, lut_type):  # pragma: no cover
    table = WORLD.value_lut if lut_type == LUTType.VALUE else WORLD.service_lut
    with WORLD.lut_lock:
        if (group_name, key) in table:
            if table[(group_name, key)] == proc_name:
                table.pop((group_name, key))
                return True
        return False


def _rpc_get_lut_entry(group_name, key, lut_type):  # pragma: no cover
    table = WORLD.value_lut if lut_type == LUTType.VALUE else WORLD.service_lut
    with WORLD.lut_lock:
        if (group_name, key) in table:
            return True, table[(group_name, key)]
        else:
            return False, None


def _rpc_has_lut_entry(group_name, key, lut_type):  # pragma: no cover
    table = WORLD.value_lut if lut_type == LUTType.VALUE else WORLD.service_lut
    with WORLD.lut_lock:
        if (group_name, key) in table:
            return True
        else:
            return False


def _rpc_call_func(func, args, kwargs):  # pragma: no cover
    # Call a function/bound method
    try:
        return func(*args, **kwargs)
    except BaseException as e:
        raise RpcException(e)


def _rpc_call_service(group_name, key, args, kwargs):  # pragma: no cover
    # call a registered service
    world = get_world()
    if group_name not in world.groups:
        # could happen if group has been destroyed on this process
        # deregister the entry from lut manager
        rpc.rpc_sync(
            world.lut_manager,
            _rpc_unset_lut_entry,
            args=(group_name, key, get_cur_name(), LUTType.SERVICE),
        )
        raise KeyError(f"Group [{group_name}], not found on Process [{get_cur_name()}]")
    lut = WORLD.groups[group_name].group_service_lut

    if key in lut:
        try:
            return lut[key](*args, **kwargs)
        except BaseException as e:
            raise RpcException(e)
    else:
        # could happen if local map is not synchronized with the
        # global map
        # deregister the entry from lut manager
        rpc.rpc_sync(
            world.lut_manager,
            _rpc_unset_lut_entry,
            args=(group_name, key, get_cur_name(), LUTType.VALUE),
        )
        raise KeyError(
            f"Service [{key}] not found on Group [{group_name}], "
            f"Process [{get_cur_name()}]"
        )


def _rpc_get_paired_value(group_name, key):  # pragma: no cover
    # get a paired value
    world = get_world()
    if group_name not in world.groups:
        # could happen if group has been destroyed on this process
        # deregister the entry from lut manager
        rpc.rpc_sync(
            world.lut_manager,
            _rpc_unset_lut_entry,
            args=(group_name, key, get_cur_name(), LUTType.VALUE),
        )
        raise KeyError(f"Group [{group_name}], not found on Process [{get_cur_name()}]")

    paired_map = WORLD.groups[group_name].group_value_lut

    if key in paired_map:
        return paired_map[key]
    else:
        # could happen if local map is not synchronized with the
        # global map
        # deregister the entry from lut manager
        rpc.rpc_sync(
            world.lut_manager,
            _rpc_unset_lut_entry,
            args=(group_name, key, get_cur_name(), LUTType.VALUE),
        )
        raise KeyError(
            f"Value with key [{key}] not found on Group [{group_name}], "
            f"Process [{get_cur_name()}]"
        )


def _world_singleton(cls):
    # Decorator used to make sure that there is only one world instance.
    def _world_singleton_wrapper(*args, **kwargs):
        global WORLD
        if WORLD is None:
            WORLD = cls(*args, **kwargs)
        else:  # pragma: no cover
            raise RuntimeError("World could only be initialized once!")
        return WORLD

    return _world_singleton_wrapper


def _torch_version_less_than(major, minor):
    t_ver = [int(v) for v in t.__version__.split(".")[0:2]]
    return t_ver < [major, minor]


def get_cur_rank():
    """
    Returns:
        Current real process rank.
    """
    if WORLD is None:  # pragma: no cover
        raise RuntimeError("Distributed environment not initialized!")
    return WORLD.rank


def get_cur_name():
    """
    Returns:
        Current real process name.
    """
    if WORLD is None:  # pragma: no cover
        raise RuntimeError("Distributed environment not initialized!")
    return WORLD.name


def get_world():  # pragma: no cover
    return WORLD


def is_world_initialized():  # pragma: no cover
    return WORLD is not None


def _is_group_ready(group_name):  # pragma: no cover
    return WORLD.group_create_signals.get(group_name, None) is False


def _unlock_group(group_name):  # pragma: no cover
    WORLD.group_create_signals[group_name] = True


def _check_executor(func):
    def wrapped(self, *args, **kwargs):
        if get_cur_name() not in self.group_members:
            raise RuntimeError(
                f"You should not execute function {func.__qualname__} when "
                "current process is not a member of the group"
            )
        return func(self, *args, **kwargs)

    return wrapped


class RpcException(Exception):  # pragma: no cover
    """
    Rpc exception class.
    """

    def __init__(self, msg):
        if isinstance(msg, str):
            # used by rpc when reraising the exception on the caller side
            super().__init__(msg)
        else:
            tb = ExceptionWithTraceback(msg).tb
            super().__init__(tb)


@_world_singleton
class World:
    """
    The distributed world.
    """

    def __init__(
        self,
        name: str,
        rank: int = -1,
        world_size: int = -1,
        init_dist: bool = True,
        init_rpc: bool = True,
        dist_backend: str = "gloo",
        dist_init_method: str = "tcp://localhost:9100",
        rpc_init_method: str = "tcp://localhost:9101",
        dist_timeout: float = 60,
        rpc_timeout: float = 60,
    ):
        """
        Args:
            name: A unique name to identify current process.
            rank: A unique rank of the current process. You do not need to specify
                it if you are using `torch.distributed.launch` or `torchelastic`
            world_size:   Size of the distributed world. You do not need to specify
                it if you are using `torch.distributed.launch` or `torchelastic`
            dist_timeout: Distributed package timeout in seconds.
            rpc_timeout:  Global rpc call timeout in seconds.
        """
        self.world_size = world_size
        self.rank = rank
        self.name = name
        self.groups = {}
        self.group_create_signals = {}

        if init_dist:
            dist.init_process_group(
                backend=dist_backend,
                init_method=dist_init_method,
                timeout=timedelta(seconds=dist_timeout),
                rank=rank,
                world_size=world_size,
            )
        if init_rpc:
            rpc.init_rpc(
                self.name,
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                    init_method=rpc_init_method, rpc_timeout=rpc_timeout
                ),
            )

        # get rank-name mapping
        self.rank_name_map = {}
        for wi in rpc._get_current_rpc_agent().get_worker_infos():
            self.rank_name_map[wi.id] = wi.name

        # Start role dispatching.
        self.started = True
        self.rpc_timeout = rpc_timeout

        # map for paired values and registered services
        self.value_lut = {}
        self.service_lut = {}
        self.lut_lock = Lock()
        self.lut_manager = self.rank_name_map[0]

    def stop(self):  # pragma: no cover
        if not self.started:
            raise RuntimeError("Cannot stop the world multiple times!")
        else:
            rpc.shutdown()

    def create_collective_group(
        self, ranks: List[int], timeout: float = 60, backend: Any = None,
    ):
        """
        Create a sub process group for collective communications. This function
        is blocking and requires that all processes in ``ranks`` to
        enter this function.

        Warning:
            Do not make collective communications call in sub-processes,
            it is unsafe.

        Args:
            ranks: Ranks of involved processes.
            timeout: Timeout of operations in the new group.
            backend: New group backend.
        Returns:
            A ``Group`` with type ``Group.COLLECTIVE``
        """
        ranks = sorted(ranks)
        group = CollectiveGroup(
            dist.new_group(ranks, timedelta(seconds=timeout), backend),
            ranks.index(self.rank) if self.rank in ranks else None,
        )
        return group

    def create_rpc_group(self, group_name: str, members: List[str]):
        """
        Create a sub process group for rpc calls. This function
        is blocking and requires that all processes in ``members`` to
        enter this function.

        Args:
            group_name: A unique group name.
            members: Members of the group.

        Returns:
            A rpc group.
        """
        if get_cur_name() not in members:  # pragma: no cover
            raise RuntimeError(
                f"Creator Process [{get_cur_name()}] not in Group [{group_name}]"
            )
        if group_name in self.groups:  # pragma: no cover
            raise RuntimeError(f"Group {group_name} already exists!")
        group = RpcGroup(group_name, members)

        # set the group
        self.groups[group_name] = group

        # temporarily set a signal
        self.group_create_signals[group_name] = False
        # wait for other members to enter
        if get_cur_name() == members[0]:
            while True:
                sleep(0.1)
                future = [
                    rpc.rpc_async(m, _is_group_ready, args=(group_name,))
                    for m in members
                ]
                for fut in future:
                    if not fut.wait():
                        break
                else:
                    future = [
                        rpc.rpc_async(m, _unlock_group, args=(group_name,))
                        for m in members
                    ]
                    for fut in future:
                        fut.wait()
                    # finish syncing all processes
                    break
        else:
            while self.group_create_signals[group_name] is not True:
                sleep(0.1)
        return group

    def get_ranks(self):
        """
        Returns:
            A list of ranks of all processes.
        """
        return list(self.rank_name_map.keys())

    def get_members(self):
        """
        Returns:
            A list of names of all processes.
        """
        return list(self.rank_name_map.values())

    def __reduce__(self):  # pragma: no cover
        raise RuntimeError("World is not picklable, create it per process!")


class CollectiveGroup:
    """
    A thin wrapper of collective communication primitives of
    ``torch.distributed``, the only difference is that ``irecv``
    now supports to recv from any src
    """

    def __init__(self, group, current_relative_rank):
        """
        Do not do it your self, use :meth:`~machin.parallel\
.distributed.World.create_collective_group` instead.
        """
        self.group = group
        self.current_rank = current_relative_rank
        self.destroyed = False

    @_copy_doc(dist.send)
    def send(self, tensor, dst, tag=0):
        return dist.send(tensor, dst, self.group, tag)

    @_copy_doc(dist.recv)
    def recv(self, tensor, src=None, tag=0):
        return dist.recv(tensor, src, self.group, tag)

    @_copy_doc(dist.isend)
    def isend(self, tensor, dst, tag=0):
        return dist.isend(tensor, dst, self.group, tag)

    def irecv(self, tensor, src=None, tag=0):  # pragma: no cover
        """
        Returns:
            An object you can call .wait() on, .wait()
            will return the source rank.
        """
        # pylint: disable=protected-access

        # Original irecv doesn't support recv from any
        # but original recv does. They are essentially
        # the same except recv have a wait() call
        dist_c10d._check_single_tensor(tensor, "tensor")
        if dist_c10d._rank_not_in_group(self.group):

            class Waiter:
                def wait(self):
                    return -1

            return Waiter()

        if self.group == dist_c10d.GroupMember.WORLD:
            dist_c10d._check_default_pg()
            pg = dist_c10d._default_pg
        else:
            pg = self.group

        if src is None:
            work = pg.recv_anysource([tensor], tag)
            if self.group == dist_c10d.GroupMember.WORLD:

                class Waiter:
                    def wait(self):
                        nonlocal work
                        work.wait()
                        return work.source_rank()

                return Waiter()
            else:

                class Waiter:
                    def wait(self):
                        nonlocal work, pg
                        work.wait()
                        src_rank = work.source_rank()
                        return dist_c10d._get_global_rank(pg, src_rank)

                return Waiter()
        else:
            if self.group == dist_c10d.GroupMember.WORLD:
                work = pg.recv([tensor], src, tag)
            else:
                group_src_rank = dist_c10d._get_group_rank(pg, src)
                work = pg.recv([tensor], group_src_rank, tag)

            class Waiter:
                def wait(self):
                    nonlocal src
                    work.wait()
                    return src

            return Waiter()

    @_copy_doc(dist.broadcast)
    def broadcast(self, tensor, src, async_op=False):
        return dist.broadcast(tensor, src, self.group, async_op)

    @_copy_doc(dist.all_reduce)
    def all_reduce(self, tensor, op=dist.ReduceOp.SUM, async_op=False):
        return dist.all_reduce(tensor, op, self.group, async_op)

    @_copy_doc(dist.reduce)
    def reduce(self, tensor, dst, op=dist.ReduceOp.SUM, async_op=False):
        return dist.reduce(tensor, dst, op, self.group, async_op)

    @_copy_doc(dist.all_gather)
    def all_gather(self, tensor_list, tensor, async_op=False):
        return dist.all_gather(tensor_list, tensor, self.group, async_op)

    @_copy_doc(dist.gather)
    def gather(self, tensor, gather_list, dst=0, async_op=False):
        return dist.gather(tensor, gather_list, dst, self.group, async_op)

    @_copy_doc(dist.scatter)
    def scatter(self, tensor, scatter_list=None, src=0, async_op=False):
        return dist.scatter(tensor, scatter_list, src, self.group, async_op)

    @_copy_doc(dist.barrier)
    def barrier(self, async_op=False):
        return dist.barrier(self.group, async_op)

    @_copy_doc(dist.broadcast_multigpu)
    def broadcast_multigpu(self, tensor_list, src, async_op=False, src_tensor=0):
        return dist.broadcast_multigpu(
            tensor_list, src, self.group, async_op, src_tensor
        )

    @_copy_doc(dist.all_reduce_multigpu)
    def all_reduce_multigpu(self, tensor_list, op=dist.ReduceOp.SUM, async_op=False):
        return dist.all_reduce_multigpu(tensor_list, op, self.group, async_op)

    @_copy_doc(dist.reduce_multigpu)
    def reduce_multigpu(
        self, tensor_list, dst, op=dist.ReduceOp.SUM, async_op=False, dst_tensor=0
    ):  # pragma: no cover
        return dist.reduce_multigpu(
            tensor_list, dst, op, self.group, async_op, dst_tensor
        )

    @_copy_doc(dist.all_gather_multigpu)
    def all_gather_multigpu(
        self, output_tensor_lists, input_tensor_list, async_op=False
    ):  # pragma: no cover
        return dist.all_gather_multigpu(
            output_tensor_lists, input_tensor_list, self.group, async_op
        )

    def destroy(self):
        """
        Destroy this collective communication group.
        """
        if not self.destroyed:
            dist.destroy_process_group(self.group)
            self.destroyed = True

    def size(self):
        """
        Returns:
            collective group size.
        """
        return dist.get_world_size(self.group)

    def __reduce__(self):  # pragma: no cover
        raise RuntimeError("Group is not picklable, create it per process!")

    def __del__(self):
        self.destroy()


# TODO:
# add the heartbeat mechanism to the lut_manager, to increase robustness


class RpcGroup:
    def __init__(self, group_name, group_members, first_create=True):
        self.group_name = group_name
        self.group_members = group_members
        self.group_value_lut = {}
        self.group_service_lut = {}
        self.destroyed = False
        self._barrier_event = Event()
        self._barrier_status = False
        if first_create and self.is_member(get_cur_name()):
            self.register(
                f"_rpc_entered_barrier_{get_cur_name()}", self._rpc_entered_barrier,
            )
            self.register(f"_rpc_exit_barrier_{get_cur_name()}", self._rpc_exit_barrier)

    @_copy_doc(rpc.rpc_sync)
    def rpc_sync(self, to: str, func: Callable, timeout=-1, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return self._rpc_normal_call(rpc.rpc_sync, to, func, timeout, args, kwargs)

    @_copy_doc(rpc.rpc_async)
    def rpc_async(self, to: str, func: Callable, timeout=-1, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return self._rpc_normal_call(rpc.rpc_async, to, func, timeout, args, kwargs)

    @_copy_doc(rpc.remote)
    def remote(self, to: str, func: Callable, timeout=-1, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return self._rpc_normal_call(rpc.remote, to, func, timeout, args, kwargs)

    @_check_executor
    def pair(self, key: Any, value: Any):
        """
        Pair a value to current process group.

        Args:
            key: A key which uniquely identifies this value in this group.
                 The name only needs to be unique for this value in this
                 group.

            value: Value to be paired.

        Raise:
            ``KeyError`` if value has already been paired.
        """
        if key in self.group_value_lut:
            raise KeyError(
                f'Value with key "{key}" already paired to Group [{self.group_name}]'
            )
        # announce the pairing
        status = rpc.rpc_sync(
            get_world().lut_manager,
            _rpc_set_lut_entry,
            args=(self.group_name, key, get_cur_name(), LUTType.VALUE),
        )
        if status:
            self.group_value_lut[key] = value
        else:
            raise KeyError(
                f'Value with key "{key}" already paired to Group [{self.group_name}]'
            )

    @_check_executor
    def unpair(self, key: Any):
        """
        Unpair a paired value from current process group. The key must be
        paired by the current process.

        Args:
            key: A key which uniquely identifies this value in this group.
                 The name only needs to be unique for this value in this
                 group.
        Raise:
            ``KeyError`` if value has not been paired.
        """
        if key not in self.group_value_lut:
            raise KeyError(
                f'Value with key "{key}" not paired to Group [{self.group_name}] '
                f"on Process[{get_cur_name()}]"
            )
        # announce the unpairing
        status = rpc.rpc_sync(
            get_world().lut_manager,
            _rpc_unset_lut_entry,
            args=(self.group_name, key, get_cur_name(), LUTType.VALUE),
        )
        if status:
            self.group_value_lut.pop(key)
        else:  # pragma: no cover
            # should never happen
            raise RuntimeError(
                f'Failed to unpair value with key "{key}" from '
                f"Group [{self.group_name}], executor is Process[{get_cur_name()}]"
            )

    def is_paired(self, key: Any):
        """
        Check whether a key has been paired to the current group.

        Args:
            key: A key which uniquely identifies this value in this group.
                 The name only needs to be unique for this value in this
                 group.
        """
        return rpc.rpc_sync(
            get_world().lut_manager,
            _rpc_has_lut_entry,
            args=(self.group_name, key, LUTType.VALUE),
        )

    def get_paired(self, key: Any):
        """
        Args:
            key: Key of the paired value, in this group.

        Returns:
            A RRef to the paired value.

        Raises:
            ``KeyError`` if not found.
        """
        if key in self.group_value_lut:
            holder = get_cur_name()
        else:
            status, holder = rpc.rpc_sync(
                get_world().lut_manager,
                _rpc_get_lut_entry,
                args=(self.group_name, key, LUTType.VALUE),
            )
            if not status:
                raise KeyError(
                    f"Value with key [{key}] not found on Group [{self.group_name}], "
                )
        return rpc.remote(holder, _rpc_get_paired_value, args=(self.group_name, key))

    @_check_executor
    def register(self, key: Any, service: Any):
        """
        Register a service to current process group.

        Args:
            key: A key which uniquely identifies this service in this group.
                 The name only needs to be unique for this service in this
                 group.
            service: Service to be registered.

        Raise:
            ``KeyError`` if service has already been registered.
        """
        if key in self.group_service_lut:
            raise KeyError(
                f'Service with key "{key}" already registered '
                f"in Group [{self.group_name}]"
            )
        # announce the pairing
        status = rpc.rpc_sync(
            get_world().lut_manager,
            _rpc_set_lut_entry,
            args=(self.group_name, key, get_cur_name(), LUTType.SERVICE),
        )
        if status:
            self.group_service_lut[key] = service
        else:
            raise KeyError(
                f'Service with key "{key}" already registered '
                f"in Group [{self.group_name}]"
            )

    @_check_executor
    def deregister(self, key: Any):
        """
        Deregister service from current process group. The key must be
        paired by the current process.

        Args:
            key: A key which uniquely identifies this value in this group.
                 The name only needs to be unique for this value in this
                 group.
        Raise:
            ``KeyError`` if srvice has not been registered.
        """
        if key not in self.group_service_lut:
            raise KeyError(
                f'Service with key "{key}" not registered '
                f"in Group [{self.group_name}] "
                f"on Process[{get_cur_name()}]"
            )
        # announce the deregistration
        status = rpc.rpc_sync(
            get_world().lut_manager,
            _rpc_unset_lut_entry,
            args=(self.group_name, key, get_cur_name(), LUTType.SERVICE),
        )
        if status:
            self.group_service_lut.pop(key)
        else:  # pragma: no cover
            # should never happen
            raise RuntimeError(
                f'Failed to deregister service with key "{key}" '
                f"from Group [{self.group_name}], "
                f"executor is Process[{get_cur_name()}]"
            )

    def is_registered(self, key: Any):
        """
        Check whether a service has been registered in the current group.

        Args:
            key: A key which uniquely identifies this service in this group.
                 The name only needs to be unique for this service in this
                 group.
        """
        return rpc.rpc_sync(
            get_world().lut_manager,
            _rpc_has_lut_entry,
            args=(self.group_name, key, LUTType.SERVICE),
        )

    def registered_sync(self, key: Any, args=(), kwargs=None):
        """
        Args:
            key: Key of the registered service, in this group.
            args: Service arguments.
            kwargs: Service keyword arguments.

        Returns:
            Result returned by the service.

        Raises:
            ``KeyError`` if service is not found.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_service_call(rpc.rpc_sync, key, args, kwargs)

    def registered_async(self, key: Any, args=(), kwargs=None):
        """
        Args:
            key: Key of the registered service, in this group.
            args: Service arguments.
            kwargs: Service keyword arguments.

        Returns:
            A future object you can call ``wait()``on.
            ``wait()`` will block the thread until execution is completed,
            and will return the result returned by the service.

        Raises:
            ``KeyError`` if service is not found.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_service_call(rpc.rpc_async, key, args, kwargs)

    def registered_remote(self, key: Any, args=(), kwargs=None):
        """
        Args:
            key: Key of the registered service, in this group.
            args: Service arguments.
            kwargs: Service keyword arguments.

        Returns:
            A RRef object pointing to the result returned by the service.

        Raises:
            ``KeyError`` if service is not found.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_service_call(rpc.remote, key, args, kwargs)

    @_check_executor
    def barrier(self):
        """
        Synchronize all members in the group, until all members have entered
        a ``barrier()`` function.

        Not thread-safe.
        """

        self._barrier_status = True
        if get_cur_name() == self.group_members[0]:
            while True:
                all_entered = all(
                    self.registered_sync(f"_rpc_entered_barrier_{m}")
                    for m in self.group_members
                )
                if not all_entered:
                    sleep(0.2)
                else:
                    break
            for m in self.group_members:
                self.registered_sync(f"_rpc_exit_barrier_{m}")
        else:
            self._barrier_event.wait()

    @_check_executor
    def destroy(self):
        """
        Destroy the rpc group.

        Note: deregistration is not considered, because they will be purged
            when any lookup fail.
        """
        if not self.destroyed:
            WORLD.groups.pop(self.group_name)
            self.destroyed = True

    def size(self):
        """
        Get the number of members in group.
        """
        return len(self.group_members)

    def is_member(self, target: str = None) -> bool:
        """
        Check whether target name is a group member.
        """
        if target is None:
            target = self.get_cur_name()
        return target in self.group_members

    def get_group_name(self) -> str:
        """
        Returns:
            Name of this group.
        """
        return self.group_name

    def get_group_members(self) -> List[str]:
        """
        Returns:
            A list of group members.
        """
        return self.group_members

    @staticmethod
    def get_cur_name() -> str:
        return get_world().name

    def _rpc_normal_call(self, rpc_method, to, func, timeout, args, kwargs):
        if not self.is_member(to):  # pragma: no cover
            raise RuntimeError("RPC target is not a member of group.")

        new_args = (func, args, kwargs)
        if _torch_version_less_than(1, 6):
            return rpc_method(to, _rpc_call_func, args=new_args)
        return rpc_method(to, _rpc_call_func, args=new_args, timeout=timeout)

    def _rpc_service_call(self, rpc_method, key, args, kwargs):
        if key in self.group_service_lut:
            holder = get_cur_name()
        else:
            status, holder = rpc.rpc_sync(
                get_world().lut_manager,
                _rpc_get_lut_entry,
                args=(self.group_name, key, LUTType.SERVICE),
            )
            if not status:
                raise KeyError(
                    f"Service with key [{key}] not found on Group [{self.group_name}], "
                )
        return rpc_method(
            holder, _rpc_call_service, args=(self.group_name, key, args, kwargs)
        )

    def _rpc_entered_barrier(self):
        return self._barrier_status

    def _rpc_exit_barrier(self):
        self._barrier_status = False
        self._barrier_event.set()
        self._barrier_event.clear()

    def __reduce__(self):  # pragma: no cover
        # returns a complete description of group
        return RpcGroup, (self.group_name, self.group_members, False)
