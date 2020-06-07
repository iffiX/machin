from time import sleep
from datetime import timedelta
from threading import Thread, local
from typing import Union, List, Tuple, Dict, Any, Callable
from torch.distributed import rpc

import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

from .election import ElectionGroupStableRpc
from .role_dispatcher import RoleDispatcherElection

WORLD = None  # type: Union[None, World]


def _copy_doc(from_func):
    """
    Used by collective group to copy documents from torch.
    """
    import io
    import sys

    def _decorator(func):
        if 'sphinx' in sys.modules:
            src_doc = from_func.__doc__
            lines = io.StringIO(src_doc)
            # remove the group line
            src_doc = "".join([line for line in lines if "group" not in line])
            func.__doc__ = src_doc
        return func
    return _decorator


def _exec_role(role):
    # role thread executor.
    local.role = role
    role.on_init()
    role.main()
    role.on_stop()


def _rpc_call_method(method, inst, *args, **kwargs):
    # Call a method function of a class, ``inst`` is a class instance
    return method(inst, *args, **kwargs)


def _rpc_call_remote_method(method, inst_rref, *args, **kwargs):
    # Call a method function of a class,
    # ``inst_rref`` is a class instance wrapped by ``RRef``
    return method(inst_rref.local_value(), *args, **kwargs)


def _rpc_call_nn_module(nn_module, *args, **kwargs):
    return nn_module(*args, **kwargs)


def _rpc_get_remote_paired_value(group_name, key):
    # TODO: dump other paired maps of the same group
    global WORLD
    paired_map = WORLD.groups[group_name].group_paired_map
    if key in paired_map:
        return paired_map[key]
    else:
        raise RpcException("""
            Failed to find key {} in the paired value map of:
            Process(rank={}, name={}) from Group {}
        """.format(key, WORLD.current_rank, WORLD.current_name, group_name))


def _world_singleton(cls, *args, **kwargs):
    # Decorator used to makesure that there is only one world instance.
    def _world_singleton_wrapper():
        global WORLD
        if WORLD is None:
            WORLD = cls(*args, **kwargs)
        else:
            raise RuntimeError("World could only be initialized once!")

    return _world_singleton_wrapper


def get_cur_rank():
    """
    Returns:
        Current real process rank.
    """
    if WORLD is None:
        raise RuntimeError("Distributed environment not initialized!")
    return WORLD.current_rank


def get_cur_role():
    """
    Returns:
        Current thread role (thread is a thread of current real process).
    """
    return local.role


class RpcException(Exception):
    """
    Rpc exception class.
    """
    pass


@_world_singleton
class World:
    """
    The distributed world.
    """
    def __init__(self,
                 world_size: int,
                 current_rank: int,
                 roles: Dict[str, Tuple[type, int]],
                 init_method: str = "tcp://localhost:9100",
                 rpc_timeout: int = 60,
                 rpc_threads: int = 4,
                 rpc_role_dispatcher: Any = None
                 ):
        """
        Args:
            world_size:   Size of distributed world.
            current_rank: A unique rank of current process.
            roles: A list of roles executed by all processes.
            init_method:  Backend initialization method.
            rpc_timeout:  Global rpc call timeout in seconds.
            rpc_threads:  Rpc recv/send thread num.
            rpc_role_dispatcher: Rpc role dispatch, by default it is
                :class:`~machin.parallel.distributed.\
RoleDispatcherElection` and uses :class:`machin.parallel.\
distributed.ElectionGroupStableRpc` as its internal election implementation.
        """
        self.world_size = world_size
        self.role_dict = roles
        # Maps role Tuple[str, int] to threads
        self.role_threads = {}

        self.current_rank = current_rank
        self.ranks = [i for i in range(world_size)]
        self.real_names = ["{}".format(i) for i in range(world_size)]
        self.groups = {}
        if rpc_role_dispatcher is not None:
            self.rpc_role_dispatcher = rpc_role_dispatcher
        else:
            role_names = list(roles.keys())
            role_nums = [val[1] for val in roles.values()]
            self.rpc_role_dispatcher = RoleDispatcherElection(
                current_rank, world_size,
                role_names, role_nums,
                ElectionGroupStableRpc(
                    name="global",
                    member_ranks=self.ranks,
                    rank=current_rank,
                    timeout=rpc_timeout
                )
            )

        # "<rank-number>" is used as the unique name.
        rpc.init_rpc("{}".format(self.current_rank),
                     rank=current_rank,
                     world_size=world_size,
                     rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                         init_method=init_method,
                         num_send_recv_threads=rpc_threads,
                         rpc_timeout=timedelta(seconds=rpc_timeout)
                     ))

        # Start role dispatching.
        self.rpc_role_dispatcher.start()
        while True:
            self.rpc_role_dispatcher.get_role_update_cond().wait()
            for role in self.rpc_role_dispatcher.get_roles():
                if role not in self.role_threads:
                    role_class = self.role_dict[role[0]][0]
                    role_thread = Thread(target=_exec_role,
                                         args=(role_class(role[1]),))
                    role_thread.start()
                    self.role_threads[role] = role_thread

    def create_collective_group(self,
                                group_name: str,
                                ranks: List[int],
                                timeout: Any = dist.default_pg_timeout,
                                backend: Any = None):
        """
        Create a sub process group for collective communications.

        Args:
            group_name: A unique group name.
            ranks: Ranks of involved processes.
            timeout: Timeout of operations in the new group.
            backend: New group backend.
        Returns:
            A ``Group`` with type ``Group.COLLECTIVE``
        """
        if group_name in self.groups:
            raise RuntimeError("Group {} already existed!".format(group_name))
        ranks = sorted(ranks)
        group = CollectiveGroup(dist.new_group(ranks, timeout, backend),
                                ranks.index(self.current_rank))
        return group

    def create_rpc_group(self, group_name: str, roles: List[Any]):
        """
        Create a sub process group for rpc calls.

        Args:
            group_name: A unique group name.
            roles: Roles of involved processes.

        Returns:
            A ``Group`` with type ``Group.RPC``
        """
        if group_name in self.groups:
            raise RuntimeError("Group {} already existed!".format(group_name))
        group = RpcGroup(group_name, roles)
        self.groups[group_name] = group
        return group

    def get_group(self, group_name: str):
        """
        Get group with name ``group_name``

        Args:
            group_name: Group name

        Returns:
            Target group
        """
        return self.groups.get(group_name, None)

    def __reduce__(self):
        raise RuntimeError("World is not pickable, create it per process!")


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

    @_copy_doc(dist.irecv)
    def irecv(self, tensor, src=None, tag=0):
        # pylint: disable=protected-access

        # Original irecv doesn't support recv from any
        # but original recv does. They are essentially
        # the same except recv have a wait() call
        dist_c10d._check_single_tensor(tensor, "tensor")
        if dist_c10d._rank_not_in_group(self.group):
            return -1

        if self.group == dist_c10d.GroupMember.WORLD:
            dist_c10d._check_default_pg()
            pg = dist_c10d._default_pg
        else:
            pg = self.group

        if src is None:
            work = pg.recv_anysource([tensor], tag)
            src_rank = work.source_rank()
            if self.group == dist_c10d.GroupMember.WORLD:
                return src_rank
            else:
                return dist_c10d._get_global_rank(pg, src_rank)
        else:
            if self.group == dist_c10d.GroupMember.WORLD:
                pg.recv([tensor], src, tag).wait()
            else:
                group_src_rank = dist_c10d._get_group_rank(pg, src)
                pg.recv([tensor], group_src_rank, tag).wait()
            return src

    @_copy_doc(dist.broadcast)
    def broadcast(self, tensor, async_op=False):
        return dist.broadcast(tensor, self.current_rank, self.group, async_op)

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

    def __reduce__(self):
        raise RuntimeError("Group is not pickable, create it per process!")

    def __del__(self):
        self.destroy()


RoleHandle = Union[str, Tuple[str, int]]


class RpcGroup:
    def __init__(self, group_name, group_roles):
        self.group_name = group_name
        self.group_roles = group_roles
        self.group_paired_map = {}
        self.destroyed = False

    def rpc_sync(self,
                 to: RoleHandle, func: Callable,
                 timeout=-1, retry=True, require_in_group=True,
                 args=(), kwargs=None):
        """
        Synchronous rpc call.

        Args:
            to: Role name, e.g.: "some_role:10".
            func: Some function.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            require_in_group: Whether require the call target to be in group.
            args: Arguments.
            kwargs: Key arguments.

        Returns:
            Function results.
        """
        return self._rpc_normal_call(rpc.rpc_sync, to, func,
                                     timeout, retry, require_in_group,
                                     args, kwargs)

    def rpc_async(self,
                  to: RoleHandle, func: Callable,
                  timeout=-1, retry=True, require_in_group=True,
                  args=(), kwargs=None):
        """
        Asynchronous rpc call.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            func: Some function.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            require_in_group: Whether require the call target to be in group.
            args: Arguments.
            kwargs: Key arguments.

        Returns:
            A rpc future object you can call ``.wait()`` on.
        """
        return self._rpc_normal_call(rpc.rpc_async, to, func,
                                     timeout, retry, require_in_group,
                                     args, kwargs)

    def rpc_remote(self,
                   to: RoleHandle, func: Callable,
                   timeout=-1, retry=True, require_in_group=True,
                   args=(), kwargs=None):
        """
        Remote rpc call.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            func: Some function.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            require_in_group: Whether require the call target to be in group.
            args: Arguments.
            kwargs: Key arguments.

        Returns:
            A ``RRef`` object.
        """
        return self._rpc_normal_call(rpc.remote, to, func,
                                     timeout, retry, require_in_group,
                                     args, kwargs)

    def rpc_register_paired(self, name: Any, value: Any):
        """
        Register a paired value to current process group.

        Args:
            name: A key which uniquely identifies this value in this group.
                 The name only needs to be unique for this value in this
                 group.

            value: Value to be registered.
        """
        self.group_paired_map[name] = value

    def rpc_get_paired(self, target: RoleHandle, name: Any,
                       timeout=-1, retry=True):
        """
        Args:
            target: Role name "some_role:10" or role tuple ("some_role", 10).
            name: Name of the paired value to get.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.

        Returns:
            An RRef to the paired value.

        Raises:
            :class:`.RpcException` if not found.
        """
        # TODO: add timeout
        del timeout

        while True:
            if not self.is_member(target):
                raise RuntimeError("Target is not a member of the group.")
            rpc_target = self._get_real_name(target)
            try:
                return rpc.remote(rpc_target,
                                  _rpc_get_remote_paired_value,
                                  args=(self.group_name, name))
            except RuntimeError:
                WORLD.rpc_role_dispatcher.notify_failure(
                    self._parse_role(target)
                )
                if not retry:
                    break
                sleep(0.1)

    def rpc_paired_class_sync(self,
                              to: RoleHandle, cls_method: Callable, name: Any,
                              timeout=-1, retry=True, require_in_group=True,
                              args=(), kwargs=None):
        """
        Call the specified ``cls_method`` on ``to`` using ``name`` to find
        the class instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            cls_method: Class method, e.g.:``some_class.some_method``
            name: Class instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            require_in_group: Whether require the call target to be in group.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_sync`
        """
        return self._rpc_paired_class_call(rpc.rpc_sync, to, cls_method, name,
                                           timeout, retry, require_in_group,
                                           args, kwargs)

    def rpc_paired_class_async(self,
                               to: RoleHandle, cls_method: Callable, name: Any,
                               timeout=-1, retry=True, require_in_group=True,
                               args=(), kwargs=None):
        """
        Call the specified ``cls_method`` on ``to`` using ``name`` to find
        the class instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            cls_method: Class method, e.g.:``some_class.some_method``
            name: Class instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            require_in_group: Whether require the call target to be in group.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_async`
        """
        return self._rpc_paired_class_call(rpc.rpc_async, to, cls_method, name,
                                           timeout, retry, require_in_group,
                                           args, kwargs)

    def rpc_paired_class_remote(self,
                                to: RoleHandle, cls_method: Callable, name: Any,
                                timeout=-1, retry=True, require_in_group=True,
                                args=(), kwargs=None):
        """
        Call the specified ``cls_method`` on ``to`` using ``name`` to find
        the class instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            cls_method: Class method, e.g.:``some_class.some_method``
            name: Class instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            require_in_group: Whether require the call target to be in group.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_remote`
        """
        return self._rpc_paired_class_call(rpc.remote, to, cls_method, name,
                                           timeout, retry, require_in_group,
                                           args, kwargs)

    def rpc_paired_nn_module_sync(self,
                                  to: RoleHandle, name: Any,
                                  timeout=-1, retry=True, require_in_group=True,
                                  args=(), kwargs=None):
        """
        Run the forward pass on ``to`` using ``name`` to find
        the model instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            name: Model instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            require_in_group: Whether require the call target to be in group.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_sync`
        """
        return self._rpc_paired_nn_module_call(rpc.rpc_sync, to, name,
                                               timeout, retry, require_in_group,
                                               args, kwargs)

    def rpc_paired_nn_module_async(self,
                                   to: RoleHandle, name: Any,
                                   timeout=-1, retry=True,
                                   require_in_group=True,
                                   args=(), kwargs=None):
        """
        Run the forward pass on ``to`` using ``name`` to find
        the model instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            name: Model instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            require_in_group: Whether require the call target to be in group.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_async`
        """
        return self._rpc_paired_nn_module_call(rpc.rpc_async, to, name,
                                               timeout, retry, require_in_group,
                                               args, kwargs)

    def rpc_paired_nn_module_remote(self,
                                    to: RoleHandle, name: Any,
                                    timeout=-1, retry=True,
                                    require_in_group=True,
                                    args=(), kwargs=None):
        """
        Run the forward pass on ``to`` using ``name`` to find
        the model instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            name: Model instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            require_in_group: Whether require the call target to be in group.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_remote`
        """
        return self._rpc_paired_nn_module_call(rpc.remote, to, name,
                                               timeout, retry, require_in_group,
                                               args, kwargs)

    def _rpc_normal_call(self, rpc_method, to, func,
                         timeout, retry, require_in_group,
                         args, kwargs):
        """
        TODO: add timeout.
        """
        del timeout

        while True:
            if require_in_group and not self.is_member(to):
                raise RuntimeError("RPC target is not a member of group.")
            rpc_to = self._get_real_name(to)
            args = list(args)
            try:
                if hasattr(func, "__self__"):
                    # instance rref, instance might be builtin or user-defined
                    args = [func, func.__self__] + args
                    return rpc_method(rpc_to, _rpc_call_method, args=args,
                                      kwargs=kwargs)
                else:
                    return rpc_method(rpc_to, func, args=args, kwargs=kwargs)
            except RuntimeError:
                WORLD.rpc_role_dispatcher.notify_failure(
                    self._parse_role(to)
                )
                if not retry:
                    break
                sleep(0.1)

    def _rpc_paired_class_call(self, rpc_method, to, cls_method, name,
                               timeout, retry, require_in_group,
                               args, kwargs):
        """
        TODO: add timeout.
        """
        del timeout

        while True:
            if require_in_group and not self.is_member(to):
                raise RuntimeError("RPC target is not a member of group.")
            rpc_to = self._get_real_name(to)
            if hasattr(cls_method, "__self__"):
                inst_rref = self.rpc_get_paired(to, name)
                args = tuple([cls_method, inst_rref] + list(args))
                try:
                    return rpc_method(rpc_to, _rpc_call_remote_method,
                                      args=args, kwargs=kwargs)
                except RuntimeError:
                    WORLD.rpc_role_dispatcher.notify_failure(
                        self._parse_role(to)
                    )
                    if not retry:
                        break
                    sleep(0.1)
            else:
                raise RuntimeError("Method does not belong to "
                                   "any registered paired class")

    def _rpc_paired_nn_module_call(self, rpc_method, to, name,
                                   timeout, retry, require_in_group,
                                   args, kwargs):
        """
        TODO: add timeout.
        """
        del timeout

        while True:
            if require_in_group and not self.is_member(to):
                raise RuntimeError("RPC target is not a member of group.")
            rpc_to = self._get_real_name(to)
            nnm_rref = self.rpc_get_paired(to, name)
            args = tuple([nnm_rref] + list(args))
            try:
                return rpc_method(rpc_to, _rpc_call_nn_module,
                                  args=args, kwargs=kwargs)
            except RuntimeError:
                WORLD.rpc_role_dispatcher.notify_failure(
                    self._parse_role(to)
                )
                if not retry:
                    break
                sleep(0.1)

    def destroy(self):
        """
        Destroy the rpc group.
        """
        if not self.destroyed:
            if len(WORLD.groups) == 1:
                rpc.shutdown(True)
                WORLD.groups.clear()
            else:
                WORLD.groups.pop(self.group_name)
            self.destroyed = True

    def size(self):
        """
        Get the number of roles in group.
        """
        return len(self.group_roles)

    def is_member(self, role: str) -> bool:
        """
        Check whether ``role`` is a group member.

        Args:
            role: A string like "some_role:10"
        """
        return self._parse_role(role) in self.group_roles

    def get_group_members(self) -> List[Tuple[str, int]]:
        """
        Returns:
            A list of group members (roles).
        """
        return self.group_roles

    @staticmethod
    def get_cur_role():
        """
        Returns:
            Current role of the executor thread.
        """
        return get_cur_role()

    @classmethod
    def _get_real_name(cls, role: str) -> str:
        # get the real rpc process name used in rpc api call
        role = cls._parse_role(role)
        return str(WORLD.rpc_role_dispatcher.get_rank(role))

    @staticmethod
    def _parse_role(role: RoleHandle) -> Tuple:
        if (isinstance(role, tuple) and len(role) == 2 and
                isinstance(role[0], str) and isinstance(role[1], int)):
            return role
        # parse a role string "some_role:10" to tuple ("some_role", 10)
        role = list(role.split(':'))
        role[1] = int(role[1])
        role = tuple(role)
        return role

    def __reduce__(self):
        raise RuntimeError("Group is not pickable, create it per process!")

    def __del__(self):
        self.destroy()
