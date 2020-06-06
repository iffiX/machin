from typing import Any, Union
from threading import Lock, Condition
from copy import deepcopy
from queue import Queue
from random import choice

import torch as t
import torch.nn as nn

from machin.parallel.distributed import RpcGroup
from machin.utils.prepare import prep_load_state_dict
from .ordered_server import OrderedServerBase, SimpleOrderedServer


class PushPullModelServer:
    """
    A simple parameter server, which synchronize model parameters
    by pushing and pulling all parameters and maintaining a strict
    ordered version chain.

    Warning:
        ``DistributedDataParallel`` is not supported.
    """
    MODEL_NAME = "_model"

    def __init__(self,
                 group: RpcGroup,
                 master: int = 0,
                 server: OrderedServerBase = None):
        """
        Args:
            group: Server group.
            master: The relative rpc rank of the master server in
                the default :class:`~machin.parallel.server.\
ordered_server.SimpleOrderedServer` implementation.
            server: Custom ordered server, mutually exclusive with
                ``master``.
        """
        self.server = (SimpleOrderedServer(group, master=master)
                       if server is None
                       else server)

    def push(self, model: nn.Module):
        """
        Try to push a model to the ordered server, if failed, the newest
        model will be automatically pulled and its parameters will be
        assigned to ``model``.

        Args:
            model: Model to push.
        """
        if not hasattr(model, "pp_version"):
            model.pp_version = 0
        model = deepcopy(model).to("cpu")

        if not self.server.push(
                self.MODEL_NAME, model.to("cpu").state_dict(),
                version=model.pp_version + 1, prev_version=model.pp_version
        ):
            result = self.server.pull(self.MODEL_NAME)
            if result is None:
                raise RuntimeError("Pull failed, this should not happen.")
            st_dict, version = result
            model.load_state_dict(st_dict)
            model.pp_version = version

    def pull(self, model: nn.Module):
        """
        Pull the newest state dict of your model and update its parameters
        and ``pp_version``.

        Args:
            model: Model to pull.
        """
        result = self.server.pull(self.MODEL_NAME)
        if result is None:
            return
        st_dict, version = result
        if not hasattr(model, "pp_version") or model.pp_version < version:
            prep_load_state_dict(model, st_dict)
            model.pp_version = version


class PushPullGradServer:
    """
    A simple parameter server, which synchronize model parameters
    by pushing gradients and pulling back new parameters, no strict
    order is guaranteed.

    Warning:
        ``DistributedDataParallel`` is not supported.
    """
    REDUCE_MASTER = 0
    REDUCE_SLAVE = 1

    def __init__(self,
                 group: RpcGroup,
                 reduce_master: int = 0,
                 model_master: int = 0,
                 server: OrderedServerBase = None,
                 reduce_device: Union[t.device, str] = "cpu",
                 reduce_batch_size: int = 64,
                 max_queue_size: int = 1024):
        """
        Note:
            Internally the master reducer will push updated versions
            to the ordered server group, this group does not provide
            ordering, and just provide model replication.

            Since a lot of workers will pull models concurrently (e.g.: A3C),
            we need to avoid the bottleneck of a single server providing
            the updated model to all these workers, therefore we need
            replication.

        Hint:
            Reduction is performed in a tree fashion:

            1. In the first step, clients will push new gradients to a
               random slave reducer, and the slave reducer will perform
               the first reduction pass, then slave reducers will push
               their results to the master reducer.
            2. In the second step, the master reducer will reduce results
               from the slave reducer to get the final reduced gradient
               dictionary (has the same structure as state_dict), and assign
               gradients to its **managed model**, and perform the
               optimization.
            3. In the final step, the master reducer will push the final
               model to the model server group, then clients can pull the
               newest model.

        Args:
            group: Server group.
            reduce_master: The relative rpc rank of the master reducer,
                which collects reduced gradients from slave reducers and
                perform the final reduction.
            model_master: The relative rpc rank of the master server in
                the default :class:`~machin.parallel.server.\
ordered_server.SimpleOrderedServer` implementation.
            server: Custom ordered server, mutually exclusive with
                ``master``.
            reduce_device: Device to perform reduction, by default it is "cpu".
            reduce_batch_size: Size of a single reduction batch, server will
                wait until the number of requests in the reduction queue have
                reached this size.
            max_queue_size: Maximum reduction request queue size.
        """
        self.server = (SimpleOrderedServer(group, master=model_master)
                       if server is None
                       else server)
        self.group = group
        self.group.rpc_register_paired(self.__class__, self)
        self.reduce_master = self.group.get_group_members()[reduce_master]
        self.reduce_slaves = self.group.get_group_members()
        self.reduce_batch_size = reduce_batch_size
        self.reduce_device = reduce_device
        self.master_queue = Queue(maxsize=max_queue_size)
        self.slave_queue = Queue(maxsize=max_queue_size)
        self.work_cond = Condition(Lock())
        self.model = None  # type: Union[nn.Module, None]
        self.optimizer = None

    def push(self, model: nn.Module):
        """
        Push the gradients of your model. Its gradients will be cleared.

        Args:
            model: Model to push.
        """
        # extract gradients from the model
        grad_dict = {}
        for k, v in model.parameters():
            grad_dict[k] = v.grad
        model.zero_grad()
        self.group.rpc_paired_class_async(
            choice(self.reduce_slaves),
            self._push_reply,
            self.__class__,
            args=(grad_dict, self.REDUCE_SLAVE)
        )

    def pull(self, model: nn.Module):
        """
        Pull the newest model.

        Args:
            model: Model to push.
        """
        params = self.server.pull("_push_pull_grad_managed_model")
        prep_load_state_dict(model, params)

    def manage_model(self, model: nn.Module, optimizer: Any):
        """
        Let the main reducer manage your model. Must be called before start.

        Args:
            model: Model to manage.
            optimizer: Optimizer of your model.

        Raises:
            ``RuntimeError`` if current rpc role is not the main reducer.
        """
        if self.group.get_cur_role() == self.reduce_master:
            self.model = model
            self.optimizer = optimizer
        else:
            raise RuntimeError("Current worker is not reduce master, and"
                               "cannot manage the model.")

    def _push_reply(self, grad_dict, level):
        # Append reduce requests to queue.
        if level == self.REDUCE_SLAVE:
            self.slave_queue.put_nowait(grad_dict)
            self.work_cond.notify_all()
        elif level == self.REDUCE_MASTER:
            self.master_queue.put_nowait(grad_dict)
            self.work_cond.notify_all()

    def _task_reduce_grad(self):
        while True:
            # Wait until one queue has reached target batch size
            self.work_cond.wait_for(
                lambda: (self.master_queue.qsize() > self.reduce_batch_size or
                         self.slave_queue.qsize() > self.reduce_batch_size)
            )
            if self.master_queue.qsize() > self.reduce_batch_size:
                # Perform reduction on the master reduction queue
                # Only the master reducer will execute this branch
                grad_dict = self._reduce_batch(self.master_queue,
                                               self.reduce_batch_size,
                                               self.reduce_device)
                # Assign gradients to the managed model and
                # perform optimization.
                if self.model is not None and self.optimizer is not None:
                    self.optimizer.zero_grad()
                    with t.no_grad():
                        for k, v in self.model.parameters():
                            v.grad = grad_dict[k].to(v.device)
                    self.optimizer.step()
                self.server.push("_push_pull_grad_managed_model",
                                 self.model.to("cpu").parameters(), 0, 0)

            elif self.slave_queue.qsize() > self.reduce_batch_size:
                # Perform reduction on the slave reduction queue
                # All processes(including master) in the reduction
                # group will execute this branch.
                grad_dict = self._reduce_batch(self.master_queue,
                                               self.reduce_batch_size,
                                               self.reduce_device)
                # Push reduced results to the master queue.
                self.group.rpc_paired_class_async(
                    self.reduce_master,
                    self._push_reply,
                    self.__class__,
                    args=(grad_dict, self.REDUCE_MASTER)
                )

    @staticmethod
    def _reduce_batch(queue, batch_size, reduce_device):
        """
        Perform batched gradient reduction

        Returns:
            Reduced gradient dictionary.
        """
        batch = []
        while len(batch) < batch_size:
            batch.append(queue.get())
        grad_dict = {}
        for grad in batch:
            for k, v in grad.items():
                if k not in grad_dict:
                    grad_dict[k] = [v.to(reduce_device)]
                else:
                    grad_dict[k].append(v.to(reduce_device))
        for k, v in grad_dict:
            # Stack parameter tensors in dim 0 and reduce.
            grad_dict[k] = t.mean(t.stack(v), dim=0, keepdim=True)
        return grad_dict
