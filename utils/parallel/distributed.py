import uuid
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

from datetime import timedelta
from typing import Union, List
from torch.distributed import rpc

world = None


def _call_method(method, inst_rref, *args, **kwargs):
    return method(inst_rref.local_value(), *args, **kwargs)


def _call_nn_module(nn_module, *args, **kwargs):
    return nn_module(*args, **kwargs)


def _get_remote_name():
    global world
    return world.current_name


def _get_remote_paired_value(group_name, key):
    # TODO: dump other paired maps of the same group
    global world
    paired_map = world.groups[group_name].group_paired_map
    if key in paired_map:
        return paired_map[key]
    else:
        raise RpcException("""
            Failed to find key {} in the paired value map of:
            Process(rank={}, name={}) from Group {}
        """.format(key, world.current_rank, world.current_name, group_name))


def _world_singleton(cls, *args, **kw):
    def _world_singleton_wrapper():
        global world
        if world is None:
            world = cls(*args, **kw)
        else:
            raise RuntimeError("World could only be initialized once!")

    return _world_singleton_wrapper


def get_cur_rank():
    return world.current_rank


def get_cur_name():
    return world.current_name


def get_cur_real_name():
    return world.current_real_name


class RpcException(Exception):
    pass


@_world_singleton
class World:
    def __init__(self,
                 world_size: int,
                 current_rank: int,
                 current_name: Union[None, str] = None,
                 backend: Union[None, str] = None,
                 init_method: str = "tcp://localhost:9100",
                 rpc_timeout: int = 600,
                 rpc_threads: int = 4
                 ):
        """
        TODO: add backend

        Args:
            world_size:   Size of distributed world.
            current_rank: A unique rank of current process.
            current_name: A unique name of current process.
            backend:      Backend type, default is "gloo", and could only be "gloo" for now.
            init_method:  Backend initialization method. could be "env://", "tcp://"...,
                          see pytorch distributed doc.
            rpc_timeout:  Global rpc call timeout in seconds, does not affect collective communications.
            rpc_threads:  Rpc recv/send thread num.
        """
        self.world_size = world_size
        self.current_rank = current_rank
        self.current_real_name = "{}".format(current_rank)
        self.current_name = self.current_real_name if current_name is None else current_name
        self.ranks = [i for i in range(world_size)]
        self.real_names = ["{}".format(i) for i in range(world_size)]
        self.names = [self.get_name(i) for i in range(world_size)]
        self.groups = {}

        rpc.init_rpc(self.current_real_name,
                     rank=current_rank,
                     world_size=world_size,
                     rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                         init_method=init_method,
                         num_send_recv_threads=rpc_threads,
                         rpc_timeout=timedelta(seconds=rpc_timeout)
                     ))

    def create_collective_group(self, group_name: str, ranks: List[int]):
        """
        Create a sub process group for collective communications.

        Args:
            group_name: A unique group name.
            ranks: Ranks of involved processes.

        Returns:
            A ``Group`` with type ``Group.COLLECTIVE``
        """
        if group_name in self.groups:
            raise RuntimeError("Group {} already existed!".format(group_name))
        ranks = sorted(ranks)
        group = Group(Group.COLLECTIVE,
                      self.current_rank,
                      None,
                      group_name,
                      self._build_map(ranks),
                      ranks)
        self.groups[group_name] = group
        return group

    def create_rpc_group(self, group_name: str, ranks: List[int]):
        """
        Create a sub process group for rpc calls.

        Args:
            group_name: A unique group name.
            ranks: Ranks of involved processes.

        Returns:
            A ``Group`` with type ``Group.RPC``
        """
        if group_name in self.groups:
            raise RuntimeError("Group {} already existed!".format(group_name))
        ranks = sorted(ranks)
        group = Group(Group.RPC,
                      self.current_rank,
                      None,
                      group_name,
                      self._build_map(ranks),
                      ranks)
        self.groups[group_name] = group
        return group

    def get_name(self, rank: int):
        """
        Get user specifier name of process with rank ``rank``

        Args:
            rank: Target process rank.

        Returns:
            Target process name.
        """
        if rank == self.current_rank:
            return self.current_name
        else:
            return rpc.rpc_sync("{}".format(rank), _get_remote_name)

    def get_group(self, group_name: str):
        """
        Get group with name ``group_name``

        Args:
            group_name: Group name

        Returns:
            Target group
        """
        return self.groups.get(group_name, None)

    def _build_map(self, ranks: List[int]):
        real_names = [self.real_names[r] for r in ranks]
        names = [self.names[r] for r in ranks]
        r_map = {}
        r_map.update({rn: r for rn, r in zip(real_names, ranks)})
        r_map.update({n: r for n, r in zip(names, ranks)})
        rela_r_map = {}
        rela_r_map.update({r: r_idx for r, r_idx in zip(ranks, range(len(ranks)))})
        rela_r_map.update({rn: r_idx for rn, r_idx in zip(real_names, range(len(ranks)))})
        rela_r_map.update({n: r_idx for n, r_idx in zip(names, range(len(ranks)))})
        rn_map = {}
        rn_map.update({r: rn for r, rn in zip(ranks, real_names)})
        rn_map.update({n: rn for n, rn in zip(names, real_names)})

        return [r_map, rela_r_map, rn_map]

    def __reduce__(self):
        raise RuntimeError("World is not pickable, create it per process!")


class Group:
    RPC = 0
    COLLECTIVE = 1

    def __init__(self, type, current_rank, group, group_name, group_map, group_ranks,
                 parent_group=None):
        """
        Please do not create Group by yourself, use ``create_collective_group`` and
        ``create_rpc_group`` api of ``World`` instead.
        """
        self.type = type
        self.group = group
        self.group_map = group_map
        self.group_name = group_name
        self.group_ranks = group_ranks
        self.current_rank = current_rank
        self.group_paired_map = {}
        self.destroyed = False
        self.parent_group = parent_group
        self.sub_counter = 0
        self.sub_groups = {}

    def _require_rpc(self, func):
        def wrapper(*args, **kwargs):
            if self.type != self.RPC:
                raise RuntimeError("This function require a rpc group to run!")
            else:
                return func(*args, **kwargs)

        return wrapper

    def _require_collective(self, func):
        def wrapper(*args, **kwargs):
            if self.type != self.COLLECTIVE:
                raise RuntimeError("This function require a collective group to run!")
            else:
                return func(*args, **kwargs)

        return wrapper

    @_require_collective
    def send(self, tensor, dst, tag=0, relative=True):
        if not relative:
            dst = self.get_rela_rank(dst)
        return dist.send(tensor, dst, self.group, tag)

    @_require_collective
    def recv(self, tensor, src=None, tag=0, relative=True):
        if src is not None and not relative:
            src = self.get_rela_rank(src)
        return dist.recv(tensor, src, self.group, tag)

    @_require_collective
    def isend(self, tensor, dst, tag=0, relative=True):
        if not relative:
            dst = self.get_rela_rank(dst)
        return dist.isend(tensor, self.get_rank(dst), self.group, tag)

    @_require_collective
    def irecv(self, tensor, src=None, tag=0, relative=True):
        # Original irecv doesn't support recv from any
        # but original recv does. They are essentially
        # the same except wait() call
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
            if not relative:
                src = self.get_rela_rank(src)
            if self.group == dist_c10d.GroupMember.WORLD:
                pg.recv([tensor], src, tag).wait()
            else:
                group_src_rank = dist_c10d._get_group_rank(pg, src)
                pg.recv([tensor], group_src_rank, tag).wait()
            return src

    @_require_collective
    def broadcast(self, tensor, async_op=False):
        return dist.broadcast(tensor, self.current_rank, self.group, async_op)

    @_require_collective
    def all_reduce(self, tensor, op=dist.ReduceOp.SUM, async_op=False):
        return dist.all_reduce(tensor, op, self.group, async_op)

    @_require_collective
    def reduce(self, tensor, dst, op=dist.ReduceOp.SUM, async_op=False, relative=True):
        if not relative:
            dst = self.get_rela_rank(dst)
        return dist.reduce(tensor, dst, op, self.group, async_op)

    @_require_collective
    def all_gather(self, tensor_list, tensor, async_op=False):
        return dist.all_gather(tensor_list, tensor, self.group, async_op)

    @_require_collective
    def gather(self, tensor, gather_list, dst=0, async_op=False, relative=True):
        if not relative:
            dst = self.get_rela_rank(dst)
        return dist.gather(tensor, gather_list, dst, self.group, async_op)

    @_require_collective
    def scatter(self, tensor, scatter_list=None, src=0, async_op=False, relative=True):
        if not relative:
            src = self.get_rela_rank(src)
        return dist.scatter(tensor, scatter_list, src, self.group, async_op)

    @_require_collective
    def barrier(self, async_op=False):
        return dist.barrier(self.group, async_op)

    @_require_rpc
    def rpc_sync(self, to, func, timeout=-1, require_in_group=True, args=(), kwargs=None):
        return self._rpc_normal_call(rpc.rpc_sync, to, func, require_in_group,
                                     timeout, args, kwargs)

    @_require_rpc
    def rpc_async(self, to, func, timeout=-1, require_in_group=True, args=(), kwargs=None):
        return self._rpc_normal_call(rpc.rpc_async, to, func, require_in_group,
                                     timeout, args, kwargs)

    @_require_rpc
    def rpc_remote(self, to, func, timeout=-1, require_in_group=True, args=(), kwargs=None):
        return self._rpc_normal_call(rpc.remote, to, func, require_in_group,
                                     timeout, args, kwargs)

    @_require_rpc
    def rpc_register_paired(self, name, value):
        """
        Register a paired value to current process group.

        Args:
            name: A key which uniquely identifies this value in this group.
                 The name only needs to be unique for this value in this
                 group.

            value: Value to be registered.
        """
        self.group_paired_map[name] = value

    @_require_rpc
    def rpc_get_paired(self, peer, name):
        return rpc.remote(self.get_real_name(peer),
                          _get_remote_paired_value,
                          args=(self.group_name, name))

    @_require_rpc
    def rpc_paired_class_sync(self, to, cls_method, name, timeout=-1, args=(), kwargs=None):
        return self._rpc_paired_class_call(rpc.rpc_sync, to, cls_method, name,
                                           timeout, args, kwargs)

    @_require_rpc
    def rpc_paired_class_async(self, to, cls_method, name, timeout=-1, args=(), kwargs=None):
        return self._rpc_paired_class_call(rpc.rpc_async, to, cls_method, name,
                                           timeout, args, kwargs)

    @_require_rpc
    def rpc_paired_class_remote(self, to, cls_method, name, timeout=-1, args=(), kwargs=None):
        return self._rpc_paired_class_call(rpc.remote, to, cls_method, name,
                                           timeout, args, kwargs)

    @_require_rpc
    def rpc_paired_nn_module_sync(self, to, name, timeout=-1, args=(), kwargs=None):
        return self._rpc_paired_nn_module_call(rpc.rpc_sync, to, name,
                                               timeout, args, kwargs)

    @_require_rpc
    def rpc_paired_nn_module_async(self, to, name, timeout=-1, args=(), kwargs=None):
        return self._rpc_paired_nn_module_call(rpc.rpc_async, to, name,
                                               timeout, args, kwargs)

    @_require_rpc
    def rpc_paired_nn_module_remote(self, to, name, timeout=-1, args=(), kwargs=None):
        return self._rpc_paired_nn_module_call(rpc.remote, to, name,
                                               timeout, args, kwargs)

    @_require_rpc
    def _rpc_normal_call(self, rpc_method, to, func, timeout, require_in_group, args, kwargs):
        """
        TODO: wait for timeout argument to be added in rpc.rpc_sync
        """
        if require_in_group and not self.is_peer(to):
            raise RuntimeError("RPC target is not a member of group.")
        to = self.get_real_name(to)
        if hasattr(func, "__self__"):
            # instance rref, instance might be builtin or user-defined
            args = tuple([func, func.__self__] + list(args))
            return rpc_method(to, _call_method, args=args, kwargs=kwargs)
        else:
            return rpc_method(to, func, args=args, kwargs=kwargs)

    @_require_rpc
    def _rpc_paired_class_call(self, rpc_method, to, cls_method, name, timeout, args, kwargs):
        """
        TODO: wait for timeout argument to be added in rpc.rpc_sync
        """
        if not self.is_peer(to):
            raise RuntimeError("RPC target is not a member of group.")
        to = self.get_real_name(to)
        if hasattr(cls_method, "__self__"):
            inst_rref = self.rpc_get_paired(to, name)
            args = tuple([cls_method, inst_rref] + list(args))
            return rpc_method(_call_method, args=args, kwargs=kwargs)
        else:
            raise RuntimeError("Method does not belong to any registered paired class")

    @_require_rpc
    def _rpc_paired_nn_module_call(self, rpc_method, to, name, timeout, args, kwargs):
        """
        TODO: wait for timeout argument to be added in rpc.rpc_sync
        """
        if not self.is_peer(to):
            raise RuntimeError("RPC target is not a member of group.")
        to = self.get_real_name(to)
        nnm_rref = self.rpc_get_paired(to, name)
        args = tuple([nnm_rref] + list(args))
        return rpc_method(_call_nn_module, args=args, kwargs=kwargs)

    def destroy(self):
        if not self.destroyed:
            if self.type == self.RPC:
                if len(world.groups) == 1:
                    rpc.shutdown(True)
                    world.groups.clear()
                else:
                    world.groups.pop(self.group_name)
                    if self.parent_group is not None:
                        self.parent_group.pop(self.group_name)
            else:
                dist.destroy_process_group(self.group)

    def size(self):
        return dist.get_world_size(self.group)

    def is_peer(self, target: Union[int, str, rpc.WorkerInfo]):
        return self.get_rank(target) in self.group_ranks

    def is_member(self):
        return world.current_rank in self.group_ranks

    @staticmethod
    def get_cur_rank():
        return world.current_rank

    @staticmethod
    def get_cur_name():
        return world.current_name

    @staticmethod
    def get_cur_real_name():
        return world.current_real_name

    def get_rank(self, target: Union[int, str, rpc.WorkerInfo]):
        if isinstance(target, rpc.WorkerInfo):
            return target.id
        elif isinstance(target, str):
            return self.group_map[0][target]
        elif isinstance(target, int):
            return target
        else:
            raise TypeError("Input must be string or int!")

    def get_rela_rank(self, target: Union[int, str, rpc.WorkerInfo]):
        if isinstance(target, rpc.WorkerInfo):
            return self.group_map[1][target.id]
        elif isinstance(target, str):
            return self.group_map[1][target]
        elif isinstance(target, int):
            return self.group_map[1][target]
        else:
            raise TypeError("Input must be string or int!")

    @_require_rpc
    def get_worker_info(self, target: Union[int, str]):
        return rpc.get_worker_info(self.get_real_name(target))

    def get_real_name(self, target: Union[int, str, rpc.WorkerInfo]):
        if isinstance(target, rpc.WorkerInfo):
            return self.group_map[2][target.id]
        elif isinstance(target, str):
            return target
        elif isinstance(target, int):
            return self.group_map[2][target]
        else:
            raise TypeError("Input must be string or int!")

    def get_peer_ranks(self):
        return self.group_ranks

    def split(self, sub_indexes: List[int], sub_name=None):
        if sub_name is None:
            sub_name = ".sub{}".format(self.sub_counter)
            self.sub_counter += 1
        sub_ranks = [self.group_ranks[idx] for idx in sub_indexes]
        group = Group(self.type,
                      self.current_rank,
                      self.group,
                      self.group_name + sub_name,
                      self.group_map,
                      sub_ranks,
                      self)
        world.groups[group.group_name] = group
        return group

    def __reduce__(self):
        raise RuntimeError("Group is not pickable, create it per process!")

    def __del__(self):
        self.destroy()
