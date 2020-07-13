from time import sleep, time
from datetime import timedelta
from threading import local
from typing import Union, List, Tuple, Dict, Any, Callable
from torch.distributed import rpc
from machin.utils.logging import fake_logger, default_logger

import inspect
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

from .election import ElectionGroupStableRpc
from .role_dispatcher import RoleDispatcherElection
from ..thread import Thread
from ..event import Event, OrEvent

WORLD = None  # type: Union[None, World]

WORLD_LOCAL = local()


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


def _exec_role(role_class, role):
    # role thread executor.
    WORLD_LOCAL.role = role
    try:
        role_inst = role_class(role[1])
        role_inst.on_init()
        role_inst.main()
        role_inst.on_stop()
    except SystemExit as e:
        if e.code == 0:
            pass
        else:  # pragma: no cover
            raise e


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


def _rpc_call_nn_module(nn_module, args, kwargs):  # pragma: no cover
    # will throw TimeoutError if timeout
    local_module = nn_module.local_value()

    try:
        return local_module(*args, **kwargs)
    except BaseException as e:
        exc = e
    raise RpcException(exc)


def _rpc_get_remote_paired_value(role, group_name, key):  # pragma: no cover
    # TODO: dump other paired maps of the same group
    global WORLD
    begin = time()
    while (role, group_name) not in WORLD.groups:
        if role not in get_cur_roles():
            # role revoked
            return
        if time() - begin >= WORLD.rpc_timeout - 0.1:
            # so that it can be retried
            raise TimeoutError("Group {} not registered on role {}, timeout"
                               .format(group_name, role))
        # wait for group to be registered
        sleep(1e-3)
    paired_map = WORLD.groups[(role, group_name)].group_paired_map

    if key in paired_map:
        return paired_map[key]
    else:
        raise RpcException("""
            Failed to find key ({}) in the paired value map of:
            Group [{}], Role[{}]
        """.format(key, group_name, role))


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


def get_cur_rank():
    """
    Returns:
        Current real process rank.
    """
    if WORLD is None:  # pragma: no cover
        raise RuntimeError("Distributed environment not initialized!")
    return WORLD.rank


def get_cur_roles():  # pragma: no cover
    """
    Returns:
        Roles executed by current real process.
    """
    return WORLD.role_dispatcher.get_roles()


def get_cur_role():  # pragma: no cover
    """
    Returns:
        Current thread role, the thread must be the main thread of the role,
        and not sub-threads.
    """
    try:
        return WORLD_LOCAL.role
    except AttributeError:
        raise RuntimeError("You should not call get_cur_role() outside your"
                           "role class, or in sub-threads started by your"
                           "role.")


def get_world():  # pragma: no cover
    return WORLD


class RpcException(Exception):  # pragma: no cover
    """
    Rpc exception class.
    """
    def __init__(self, msg):
        if isinstance(msg, str):
            super(RpcException, self).__init__(msg)
        elif isinstance(msg, BaseException):
            super(RpcException, self).__init__(str(msg))


RoleHandle = Union[str, Tuple[str, int]]


@_world_singleton
class World:
    """
    The distributed world.
    """
    def __init__(self,
                 world_size: int,
                 rank: int,
                 roles: Dict[str, Tuple[type, int]],
                 init_method: str = "tcp://localhost:9100",
                 rpc_timeout: float = 60,
                 rpc_threads: int = 8,
                 election_group: Any = None,
                 role_dispatcher: Any = None,
                 election_timeout: float = 1e-1,
                 auto_restart: bool = True,
                 logging: bool = False
                 ):
        """
        Args:
            world_size:   Size of the distributed world,
                total number of processes in the beginning.
            rank: A unique rank of the current process.
            roles: A list of roles executed by all processes.
            init_method:  Backend initialization method.
            rpc_timeout:  Global rpc call timeout in seconds.
            rpc_threads:  Rpc recv/send thread num.
            election_group: Election group.
            role_dispatcher: Role dispatcher, by default it is
                :class:`~machin.parallel.distributed.\
RoleDispatcherElection` and uses :class:`machin.parallel.\
distributed.ElectionGroupStableRpc` as its internal election implementation.
            election_timeout: The default election timeout value
                for the default election group used by the default
                ``role_dispatcher``.
            logging: Whether to enable logging
        """
        self.world_size = world_size
        self.role_dict = roles
        # Maps role Tuple[str, int] to threads
        self.role_threads = {}

        self.rank = rank
        self.ranks = [i for i in range(world_size)]
        self.real_names = ["{}".format(i) for i in range(world_size)]
        self.groups = {}

        if logging:
            self.logger = default_logger
        else:
            self.logger = fake_logger

        if election_group is not None:  # pragma: no cover
            self.election_group = election_group
        else:
            self.election_group = ElectionGroupStableRpc(
                name="global",
                member_ranks=self.ranks,
                rank=rank,
                timeout=election_timeout,
                logging=logging
            )

        if role_dispatcher is not None:  # pragma: no cover
            self.role_dispatcher = role_dispatcher
        else:
            role_names = list(roles.keys())
            role_counts = [val[1] for val in roles.values()]
            self.role_dispatcher = RoleDispatcherElection(
                name="world_dispatcher",
                rank=rank, world_size=world_size,
                roles=role_names, role_counts=role_counts,
                election_group=self.election_group,
                logging=logging
            )

        # "<rank-number>" is used as the unique name.
        rpc.init_rpc("{:d}".format(self.rank),
                     rank=rank,
                     world_size=world_size,
                     rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                         init_method=init_method,
                         num_send_recv_threads=rpc_threads,
                         rpc_timeout=timedelta(seconds=rpc_timeout)
                     ))

        # Start role dispatching.
        self.started = True
        self.auto_restart = auto_restart
        self.stop_event = Event()
        self.run_disp_thread = Thread(target=self._task_run_dispatched_roles)
        self.run_disp_thread.start()
        self.role_dispatcher.start()
        self.rpc_timeout = rpc_timeout

    def stop(self):  # pragma: no cover
        """
        Normally you should not call this, since currently there is no
        good way to stop dispatched role threads, and python will hang
        if any role thread has not exited.
        """
        if not self.started:
            raise RuntimeError("Cannot stop the world multiple times!")
        else:
            self.stop_event.set()
            self.run_disp_thread.join()
            self.role_dispatcher.stop()
            self.election_group.stop()
            try:
                rpc.shutdown()
            except RuntimeError:
                pass

    def watch(self):
        self.run_disp_thread.watch()
        for t in self.role_threads.values():
            t.watch()
        self.role_dispatcher.watch()
        self.election_group.watch()

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

    def create_rpc_group(self, group_name: str, roles: List[Any]):
        """
        Create a sub process group for rpc calls.

        Args:
            group_name: A unique group name.
            roles: Roles of involved processes.

        Returns:
            A ``Group`` with type ``Group.RPC``
        """
        role = get_cur_role()
        if (role, group_name) in self.groups:  # pragma: no cover
            raise RuntimeError("Group {} already exists!".format(group_name))
        if role not in roles:  # pragma: no cover
            raise RuntimeError("Current role is not a part of roles of "
                               "your rpc group!")
        group = RpcGroup(role, group_name, roles)
        self.groups[(get_cur_role(), group_name)] = group
        return group

    def get_rpc_group(self, group_name: str, role: RoleHandle = None):
        """
        Get group with name ``group_name``, and created by ``role``.

        Note:
            If ``role`` is not specified, will use :func: `.get_cur_role` to
            resolve the ``role``.

            If you are calling this function from a sub-thread started by
            your role, you must specify ``role``, otherwise a ``RuntimeError``
            will be raised.

        Note:
            If ``role`` is not allocated to this process, a rpc request
            will be performed to query the process allocated with the target
            role for the group, you may perform rpc calls on the returned
            group, but all register paired values on the returned group
            are not effective. The ``cur_role`` property of the returned
            group will be set to ``(None, -1)``

            This behavior could be useful if you are trying to access a
            value/service in the target rpc group from a member outside
            the rpc group.

        Args:
            group_name: Group name

        Returns:
            Target group, ``None`` if not found
        """
        if role is None:
            role = get_cur_role()
        if role in get_cur_roles():
            return self.groups.get((role, group_name), None)
        else:
            target_process = str(self.role_dispatcher.get_rank(role))
            remote_world = rpc.remote(target_process,
                                      func=get_world())
            return rpc.rpc_sync(target_process,
                                _rpc_call_remote_method,
                                args=(World.get_rpc_group, remote_world,
                                      (group_name, role), {}))

    def _task_run_dispatched_roles(self):
        dispatcher = self.role_dispatcher
        event = OrEvent(self.stop_event, dispatcher.get_role_update_event())
        while True:
            event.wait()
            if self.stop_event.is_set():
                break
            for role in dispatcher.get_roles():
                # role: Tuple[str, int]
                # str is the role name,
                # int is the role index
                if role not in self.role_threads:
                    role_class = self.role_dict[role[0]][0]
                    role_thread = Thread(target=_exec_role,
                                         args=(role_class, role))
                    role_thread.start()
                    self.role_threads[role] = role_thread
                elif not self.role_threads[role].is_alive():
                    if self.role_threads[role].exception is not None:
                        self.logger.warning(
                            "Role {} have exited with exception:"
                            "{}"
                            .format(
                                role,
                                str(self.role_threads[role].exception)
                            )
                        )
                        if self.auto_restart:
                            self.logger.warning("Restart Role {}".format(role))
                            role_class = self.role_dict[role[0]][0]
                            self.role_threads[role].join()
                            role_thread = Thread(target=_exec_role,
                                                 args=(role_class, role))
                            role_thread.start()
                        else:
                            raise self.role_threads[role].exception

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

    @_copy_doc(dist.irecv)
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
    def __init__(self, cur_role, group_name, group_roles):
        self.cur_role = cur_role
        self.group_name = group_name
        self.group_roles = group_roles
        self.group_paired_map = {}
        self.destroyed = False

    def rpc_sync(self,
                 to: RoleHandle, func: Callable,
                 timeout=-1, retry=True, args=(), kwargs=None):
        """
        Synchronous rpc call.

        Args:
            to: Role name, e.g.: "some_role:10".
            func: Some function.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            args: Arguments.
            kwargs: Key arguments.

        Returns:
            Function results.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_normal_call(rpc.rpc_sync, to, func,
                                     timeout, retry, args, kwargs)

    def rpc_async(self,
                  to: RoleHandle, func: Callable,
                  timeout=-1, retry=True, args=(), kwargs=None):
        """
        Asynchronous rpc call.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            func: Some function.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            args: Arguments.
            kwargs: Key arguments.

        Returns:
            A rpc future object you can call ``.wait()`` on.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_normal_call(rpc.rpc_async, to, func,
                                     timeout, retry, args, kwargs)

    def rpc_remote(self,
                   to: RoleHandle, func: Callable,
                   timeout=-1, retry=True, args=(), kwargs=None):
        """
        Remote rpc call.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            func: Some function.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            args: Arguments.
            kwargs: Key arguments.

        Returns:
            A ``RRef`` object.
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_normal_call(rpc.remote, to, func,
                                     timeout, retry, args, kwargs)

    def rpc_register_paired(self, name: Any, value: Any):
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
        target = self._parse_role(target)

        if not self.is_member(target):
            raise RuntimeError("Target is not a member of the group.")
        while True:
            rpc_target = self._get_real_name(target)

            try:
                return rpc.remote(rpc_target,
                                  _rpc_get_remote_paired_value,
                                  args=(target, self.group_name, name))
            except RuntimeError:
                WORLD.role_dispatcher.notify_failure(
                    self._parse_role(target)
                )
                if not retry:
                    break
                sleep(0.1)

    def rpc_paired_class_sync(self,
                              to: RoleHandle, cls_method: Callable, name: Any,
                              timeout=-1, retry=True, args=(), kwargs=None):
        """
        Call the specified ``cls_method`` on ``to`` using ``name`` to find
        the class instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            cls_method: Class method, e.g.:``some_class.some_method``
            name: Class instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_sync`
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_class_call(rpc.rpc_sync, to, cls_method, name,
                                           timeout, retry, args, kwargs)

    def rpc_paired_class_async(self,
                               to: RoleHandle, cls_method: Callable, name: Any,
                               timeout=-1, retry=True, args=(), kwargs=None):
        """
        Call the specified ``cls_method`` on ``to`` using ``name`` to find
        the class instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            cls_method: Class method, e.g.:``some_class.some_method``
            name: Class instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_async`
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_class_call(rpc.rpc_async, to, cls_method, name,
                                           timeout, retry, args, kwargs)

    def rpc_paired_class_remote(self,
                                to: RoleHandle, cls_method: Callable, name: Any,
                                timeout=-1, retry=True, args=(), kwargs=None):
        """
        Call the specified ``cls_method`` on ``to`` using ``name`` to find
        the class instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            cls_method: Class method, e.g.:``some_class.some_method``
            name: Class instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_remote`
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_class_call(rpc.remote, to, cls_method, name,
                                           timeout, retry, args, kwargs)

    def rpc_paired_nn_module_sync(self,
                                  to: RoleHandle, name: Any,
                                  timeout=-1, retry=True, args=(), kwargs=None):
        """
        Run the forward pass on ``to`` using ``name`` to find
        the model instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            name: Model instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_sync`
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_nn_module_call(rpc.rpc_sync, to, name,
                                               timeout, retry, args, kwargs)

    def rpc_paired_nn_module_async(self,
                                   to: RoleHandle, name: Any,
                                   timeout=-1, retry=True,
                                   args=(), kwargs=None):
        """
        Run the forward pass on ``to`` using ``name`` to find
        the model instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            name: Model instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_async`
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_nn_module_call(rpc.rpc_async, to, name,
                                               timeout, retry, args, kwargs)

    def rpc_paired_nn_module_remote(self,
                                    to: RoleHandle, name: Any,
                                    timeout=-1, retry=True,
                                    args=(), kwargs=None):
        """
        Run the forward pass on ``to`` using ``name`` to find
        the model instance.

        Args:
            to: Role name "some_role:10" or role tuple ("some_role", 10).
            name: Model instance name.
            timeout: Call timeout.
            retry: Whether to retry until success after timeout.
            args: Arguments.
            kwargs: Key arguments.

        See Also:
            :meth:`.RpcGroup.rpc_remote`
        """
        if kwargs is None:
            kwargs = {}
        return self._rpc_paired_nn_module_call(rpc.remote, to, name,
                                               timeout, retry, args, kwargs)

    def _rpc_normal_call(self, rpc_method, to, func,
                         timeout, retry, args, kwargs):
        """
        TODO: add timeout.
        """
        del timeout

        while True:
            if not self.is_member(to):
                raise RuntimeError("RPC target is not a member of group.")
            rpc_to = self._get_real_name(to)
            new_args = (func, args, kwargs)
            self._log("Begin rpc normal call(func={}) to Role {}"
                      .format(func.__qualname__, to))
            try:
                return rpc_method(rpc_to, _rpc_call_func, args=new_args)
            except RuntimeError as _:
                self._log_err(
                    "Rpc normal call(func={}) to Role {} failed"
                    .format(func.__qualname__, to)
                )
                WORLD.role_dispatcher.notify_failure(
                    self._parse_role(to)
                )
            if not retry:
                break
            sleep(0.1)

    def _rpc_paired_class_call(self, rpc_method, to, cls_method, name,
                               timeout, retry, args, kwargs):
        """
        TODO: add timeout.
        """
        del timeout

        while True:
            if not self.is_member(to):
                raise RuntimeError("RPC target is not a member of group.")
            rpc_to = self._get_real_name(to)
            cls_method = self._get_real_class_method(cls_method)
            inst_rref = self.rpc_get_paired(to, name)
            new_args = (cls_method, inst_rref, args, kwargs)
            self._log("Begin rpc paired class call"
                      "(method={}, name={}, cls_method={}) to Role {}"
                      .format(rpc_method.__qualname__, name,
                              cls_method.__qualname__, to))
            try:
                return rpc_method(rpc_to, _rpc_call_remote_method,
                                  args=new_args)
            except (RuntimeError, TimeoutError) as _:
                self._log_err(
                    "Rpc paired class call to Role {} failed".format(to)
                )
                WORLD.role_dispatcher.notify_failure(
                    self._parse_role(to)
                )

            if not retry:
                break
            sleep(0.1)

    def _rpc_paired_nn_module_call(self, rpc_method, to, name,
                                   timeout, retry, args, kwargs):
        """
        TODO: add timeout.
        """
        del timeout

        while True:
            if not self.is_member(to):
                raise RuntimeError("RPC target is not a member of group.")
            rpc_to = self._get_real_name(to)
            nnm_rref = self.rpc_get_paired(to, name)
            new_args = (nnm_rref, args, kwargs)

            self._log("Begin rpc nn module call (nn_name={}) to Role {}"
                      .format(name, to))
            try:
                return rpc_method(rpc_to, _rpc_call_nn_module,
                                  args=new_args)
            except (RuntimeError, TimeoutError) as _:
                self._log_err(
                    "Rpc nn module call to Role {} failed".format(to)
                )
                WORLD.role_dispatcher.notify_failure(
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
            if self.cur_role[0] is not None:
                WORLD.groups.pop(self.cur_role, self.group_name)
            self.destroyed = True

    def size(self):
        """
        Get the number of roles in group.
        """
        return len(self.group_roles)

    def is_member(self, role: RoleHandle) -> bool:
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

    def get_cur_role(self):
        return self.cur_role

    def _log(self, msg):
        WORLD.logger.info("Rpc group<{}> Process[{}]: {}"
                          .format(self.group_name, get_cur_rank(), msg))

    def _log_err(self, msg):
        WORLD.logger.error("Rpc group<{}> Process[{}]: {}"
                           .format(self.group_name, get_cur_rank(), msg))

    @classmethod
    def _get_real_name(cls, role: RoleHandle) -> str:
        # get the real rpc process name used in rpc api call
        role = cls._parse_role(role)
        return str(WORLD.role_dispatcher.get_rank(role))

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

    @staticmethod
    def _get_real_class_method(method):
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
        return RpcGroup, ((None, -1), self.group_name, self.group_roles)

    def __del__(self):
        self.destroy()
