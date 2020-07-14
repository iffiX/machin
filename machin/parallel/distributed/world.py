from time import sleep, time
from datetime import timedelta
from typing import Union, List, Any, Callable
from torch.distributed import rpc

import inspect
import torch as t
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

WORLD = None  # type: Union[None, World]


def _copy_doc(from_func):
    """
    Used by collective group to copy documents from torch.
    """
    import io
    import sys

    def _decorator(func):
        if 'sphinx' in sys.modules:  # pragma: no cover
            src_doc = from_func.__doc__
            lines = io.StringIO(src_doc)
            # remove the group line
            src_doc = "".join([line for line in lines if "group" not in line])
            func.__doc__ = src_doc
        return func

    return _decorator


def _rpc_call_func(func, args, kwargs):  # pragma: no cover
    # Call a function/bound method
    try:
        return func(*args, **kwargs)
    except BaseException as e:
        exc = e
    raise RpcException(exc)


def _rpc_call_remote_method(method, inst_rref,
                            args, kwargs):  # pragma: no cover
    # Call a method function of a class,
    # ``inst_rref`` is a class instance wrapped by ``RRef``

    # will throw TimeoutError if timeout
    local_value = inst_rref.local_value()

    try:
        return method(local_value, *args, **kwargs)
    except BaseException as e:
        exc = e
    raise RpcException(exc)


def _rpc_call_model(model, args, kwargs):  # pragma: no cover
    # will throw TimeoutError if timeout
    local_module = model.local_value()

    try:
        return local_module(*args, **kwargs)
    except BaseException as e:
        exc = e
    raise RpcException(exc)


def _rpc_get_remote_paired_value(group_name, key):  # pragma: no cover
    global WORLD
    begin = time()
    while group_name not in WORLD.groups:
        if time() - begin >= WORLD.rpc_timeout - 0.1:
            # so that it can be retried
            raise TimeoutError("Group [{}] not registered on process [{}], "
                               "timeout".format(group_name, WORLD.rank))
        # wait for group to be registered
        sleep(1e-2)
    paired_map = WORLD.groups[group_name].group_paired_map

    if key in paired_map:
        return paired_map[key]
    else:
        raise RpcException("""
            Failed to find key ({}) in the paired value map of Group [{}].\n
            Existing map is:\n
            {}
        """.format(key, group_name, paired_map))


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


def _torch_version_less_than(major, minor, patch):
    t_ver = [int(v) for v in t.__version__.split(".")]
    return t_ver < [major, minor, patch]


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


def _get_rpc_group(group_name):  # pragma: no cover
    return WORLD.groups.get(group_name, None)


class RpcException(Exception):  # pragma: no cover
    """
    Rpc exception class.
    """

    def __init__(self, msg):
        if isinstance(msg, str):
            super(RpcException, self).__init__(msg)
        elif isinstance(msg, BaseException):
            super(RpcException, self).__init__(str(msg))


@_world_singleton
class World:
    """
    The distributed world.
    """

    def __init__(self,
                 world_size: int,
                 rank: int,
                 name: str,
                 init_method: str = "tcp://localhost:9100",
                 rpc_timeout: float = 60,
                 rpc_threads: int = 8):
        """
        Args:
            world_size:   Size of the distributed world,
                total number of processes in the beginning.
            rank: A unique rank of the current process.
            name: A unique name to identify current process.
            init_method:  Backend initialization method.
            rpc_timeout:  Global rpc call timeout in seconds.
            rpc_threads:  Rpc recv/send thread num.
        """
        self.world_size = world_size
        self.rank = rank
        self.name = name
        self.groups = {}

        # "<rank-number>" is used as the unique name.
        rpc.init_rpc(self.name,
                     rank=rank,
                     world_size=world_size,
                     rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                         init_method=init_method,
                         num_send_recv_threads=rpc_threads,
                         rpc_timeout=timedelta(seconds=rpc_timeout)
                     ))

        # Start role dispatching.
        self.started = True
        self.rpc_timeout = rpc_timeout

    def stop(self):  # pragma: no cover
        if not self.started:
            raise RuntimeError("Cannot stop the world multiple times!")
        else:
            rpc.shutdown()

    def create_collective_group(self,
                                ranks: List[int],
                                timeout: Any = dist.default_pg_timeout,
                                backend: Any = None):
        """
        Create a sub process group for collective communications. This function
        is blocking and requires that all processes in the world enter this
        function.

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
        group = CollectiveGroup(dist.new_group(ranks, timeout, backend),
                                ranks.index(self.rank))
        return group

    def create_rpc_group(self, group_name: str, members: List[str]):
        """
        Create a sub process group for rpc calls.

        Args:
            group_name: A unique group name.
            members: Members of the group.

        Returns:
            A rpc group.
        """
        if group_name in self.groups:  # pragma: no cover
            raise RuntimeError("Group {} already exists!".format(group_name))
        group = RpcGroup(group_name, members, True)
        self.groups[group_name] = group
        return group

    def get_rpc_group(self, group_name: str, target: str = None):
        """
        Get group with name ``group_name``, supports group not created
        on this process.

        Args:
            group_name: Group name.
            target: Target process used to query for the group info
                if it is not created locally, by default it is set
                to the local process.

        Returns:
            Target group, ``None`` if not found
        """

        if target is None:
            return self.groups.get(group_name, None)
        else:
            return rpc.rpc_sync(target,
                                _get_rpc_group,
                                args=(group_name,))

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


# Since roles are executed by threads scattered to all processes,
# we use (get_cur_role(), group_name) as a unique key to identify groups
# in the global WORLD.groups map, the first key is used to prevent:
#
#   Roles from the same RpcGroup are assigned to the same rpc process, then
#   their group_paired_map will overlap if we only use "group_name" as a unique
#   key in the global WORLD.groups map


class RpcGroup:
    def __init__(self, group_name, group_members, is_local):
        self.group_name = group_name
        self.group_members = group_members
        self.group_paired_map = {}
        self.is_local = is_local
        self.destroyed = False

    @_copy_doc(rpc.rpc_sync)
    def rpc_sync(self, to: str, func: Callable,
                 timeout=-1, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return self._rpc_normal_call(rpc.rpc_sync, to, func,
                                     timeout, args, kwargs)

    @_copy_doc(rpc.rpc_async)
    def rpc_async(self, to: str, func: Callable,
                  timeout=-1, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return self._rpc_normal_call(rpc.rpc_async, to, func,
                                     timeout, args, kwargs)

    @_copy_doc(rpc.remote)
    def remote(self, to: str, func: Callable,
               timeout=-1, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return self._rpc_normal_call(rpc.remote, to, func,
                                     timeout, args, kwargs)

    def rpc_pair(self, name: Any, value: Any):
        """
        Register a paired value to current process group.

        Note:
            Value will be overwritten if the name is the same.

        Args:
            name: A key which uniquely identifies this value in this group.
                 The name only needs to be unique for this value in this
                 group.

            value: Value to be registered.
        """
        self.group_paired_map[name] = value

    def rpc_get_paired(self, target: str, name: Any, timeout=-1):
        """
        Args:
            target: Target process name.
            name: Name of the paired value.
            timeout: Call timeout.

        Returns:
            An RRef to the paired value.

        Raises:
            :class:`.RpcException` if not found.
        """
        if _torch_version_less_than(1, 6, 0):
            return rpc.remote(target,
                              _rpc_get_remote_paired_value,
                              args=(self.group_name, name))

        return rpc.remote(target,
                          _rpc_get_remote_paired_value,
                          args=(self.group_name, name), timeout=timeout)

    def rpc_paired_class_sync(self,
                              to: str, name: Any, cls_method: Callable,
                              timeout=-1, args=(), kwargs=None):
        """
        Call the specified ``cls_method`` on ``to`` using ``name`` to find
        the class instance.

        Args:
            to: Target process name.
            name: Registered paired class instance name.
            cls_method: Class method, e.g.:``some_class.some_method``.
            timeout: Call timeout.
            args: Arguments.
            kwargs: Key arguments.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_class_call(rpc.rpc_sync, to, name, cls_method,
                                           timeout, args, kwargs)

    def rpc_paired_class_async(self,
                               to: str, name: Any, cls_method: Callable,
                               timeout=-1, args=(), kwargs=None):
        """
        Call the specified ``cls_method`` on ``to`` using ``name`` to find
        the class instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            cls_method: Class method, e.g.:``some_class.some_method``
            name: Class instance name.
            timeout: Call timeout.
            args: Arguments.
            kwargs: Key arguments.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_class_call(rpc.rpc_async, to, name, cls_method,
                                           timeout, args, kwargs)

    def rpc_paired_class_remote(self,
                                to: str, name: Any, cls_method: Callable,
                                timeout=-1, args=(), kwargs=None):
        """
        Call the specified ``cls_method`` on ``to`` using ``name`` to find
        the class instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            cls_method: Class method, e.g.:``some_class.some_method``
            name: Class instance name.
            timeout: Call timeout.
            args: Arguments.
            kwargs: Key arguments.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_class_call(rpc.remote, to, name, cls_method,
                                           timeout, args, kwargs)

    def rpc_paired_model_sync(self,
                              to: str, name: Any,
                              timeout=-1, args=(), kwargs=None):
        """
        Run the forward pass on ``to`` using ``name`` to find
        the model instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            name: Model instance name.
            timeout: Call timeout.
            args: Arguments.
            kwargs: Key arguments.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_model_call(rpc.rpc_sync, to, name,
                                           timeout, args, kwargs)

    def rpc_paired_model_async(self,
                               to: str, name: Any,
                               timeout=-1, args=(), kwargs=None):
        """
        Run the forward pass on ``to`` using ``name`` to find
        the model instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            name: Model instance name.
            timeout: Call timeout.
            args: Arguments.
            kwargs: Key arguments.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_model_call(rpc.rpc_async, to, name,
                                           timeout, args, kwargs)

    def rpc_paired_model_remote(self,
                                to: str, name: Any,
                                timeout=-1, args=(), kwargs=None):
        """
        Run the forward pass on ``to`` using ``name`` to find
        the model instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            name: Model instance name.
            timeout: Call timeout.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_remote`
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_model_call(rpc.remote, to, name,
                                           timeout, args, kwargs)

    def _rpc_normal_call(self, rpc_method, to, func, timeout, args, kwargs):
        if not self.is_member(to):  # pragma: no cover
            raise RuntimeError("RPC target is not a member of group.")

        new_args = (func, args, kwargs)
        if _torch_version_less_than(1, 6, 0):
            return rpc_method(to, _rpc_call_func, args=new_args)
        return rpc_method(to, _rpc_call_func, args=new_args, timeout=timeout)

    def _rpc_paired_class_call(self, rpc_method, to, name, cls_method,
                               timeout, args, kwargs):
        if not self.is_member(to):  # pragma: no cover
            raise RuntimeError("RPC target is not a member of group.")

        cls_method = self._get_real_class_method(cls_method)
        inst_rref = self.rpc_get_paired(to, name)
        new_args = (cls_method, inst_rref, args, kwargs)
        if _torch_version_less_than(1, 6, 0):
            return rpc_method(to, _rpc_call_remote_method, args=new_args)
        return rpc_method(to, _rpc_call_remote_method, args=new_args,
                          timeout=timeout)

    def _rpc_paired_model_call(self, rpc_method, to, name,
                               timeout, args, kwargs):
        if not self.is_member(to):  # pragma: no cover
            raise RuntimeError("RPC target is not a member of group.")

        m_rref = self.rpc_get_paired(to, name)
        new_args = (m_rref, args, kwargs)
        if _torch_version_less_than(1, 6, 0):
            return rpc_method(to, _rpc_call_model, args=new_args)
        return rpc_method(to, _rpc_call_model, args=new_args,
                          timeout=timeout)

    def destroy(self):
        """
        Destroy the rpc group.
        """
        if not self.destroyed:
            if self.is_local:
                WORLD.groups.pop(self.group_name)
            self.destroyed = True

    def size(self):
        """
        Get the number of members in group.
        """
        return len(self.group_members)

    def is_member(self, target: str) -> bool:
        """
        Check whether target name is a group member.
        """
        return target in self.group_members

    def get_group_members(self) -> List[str]:
        """
        Returns:
            A list of group members.
        """
        return self.group_members

    @staticmethod
    def get_cur_name() -> str:
        return get_world().name

    @staticmethod
    def _get_real_class_method(method):  # pragma: no cover
        # suppose class A has a method "func1" and a class method "func2"
        # suppose a is an instance of class A
        # then:
        # A.func1 is a function
        # A.func2 is a bound method with __self__ set to class A
        # a.func1 is a bound method with __self__ set to instance a
        # a.func2 is a bound method with __self__ set to class A
        if inspect.ismethod(method):
            return method.__func__
        else:
            return method

    def __reduce__(self):  # pragma: no cover
        # returns a complete description of group
        return RpcGroup, (self.group_name, self.group_members, False)

    def __del__(self):
        self.destroy()
