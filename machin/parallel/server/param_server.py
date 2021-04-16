from typing import Any, Union, List
from random import choice
from copy import deepcopy
from queue import Queue
from threading import Event

import enum
import torch as t
import torch.nn as nn

from machin.parallel.thread import Thread
from machin.parallel.distributed import RpcGroup
from machin.utils.prepare import prep_load_state_dict
from .ordered_server import (
    OrderedServerBase,
    OrderedServerSimple,
    OrderedServerSimpleImpl,
)


class PushPullModelServer:
    def __init__(self, model_name: str, o_server: OrderedServerBase = None):
        """
        Create an accessor to the services provided by
        :class:`PushPullModelServerImpl`

        Args:
            model_name: Name of the managed model in the ordered server,
                only needed if ``server`` needs such a identifier. The default
                ordered server does not require this.
            o_server: Ordered server accessor.
        """
        self.model_name = model_name
        self.o_server = o_server

    def push(self, model: nn.Module, pull_on_fail=True):
        """
        Try to push a model to the ordered server, if failed, the newest
        model will be automatically pulled and its parameters will be
        assigned to ``model``. Gradients will not be cleared.

        Args:
            model: Model to push.
            pull_on_fail: Pull the newest parameters if push failed.

        Returns:
            True if push succeeded, else False.
        """
        if not hasattr(model, "pp_version"):
            model.pp_version = 0

        copied_model_params = deepcopy(model.state_dict())
        for k, v in copied_model_params.items():
            copied_model_params[k] = v.to("cpu")
        if not self.o_server.push(
            self.model_name,
            copied_model_params,
            version=model.pp_version + 1,
            prev_version=model.pp_version,
        ):
            if pull_on_fail:
                result = self.o_server.pull(self.model_name)
                if result is None:  # pragma: no cover
                    raise RuntimeError("Pull failed, this should not happen.")
                st_dict, version = result
                prep_load_state_dict(model, st_dict)
                model.pp_version = version
            return False
        else:
            model.pp_version += 1
        return True

    def pull(self, model: nn.Module):
        """
        Pull the newest state dict of your model and update its parameters
        and ``pp_version``. Gradients will not be cleared.

        Args:
            model: Model to pull.

        Returns:
            True if pull succeeded, else False.
        """
        result = self.o_server.pull(self.model_name)
        if result is None:  # pragma: no cover
            return False
        st_dict, version = result
        if not hasattr(model, "pp_version") or model.pp_version < version:
            prep_load_state_dict(model, st_dict)
            model.pp_version = version
        return True


class PushPullModelServerImpl:
    """
    A simple parameter server, which synchronize model parameters
    by pushing and pulling all parameters and maintaining a strict
    ordered version chain.

    Warning:
        Only one model is supported.
    """

    def __init__(
        self,
        server_name: str,
        group: RpcGroup,
        model_name: str = "model",
        o_server: OrderedServerBase = None,
    ):
        """
        This init function must be only invoked on the runner process,
        and the runner process must be a member process of ``group``.

        Args:
            server_name: Name of this server, used to registered
                the server as a paired class of ``group``.
            group: RpcGroup of the default server :class:`.OrderedServerSimple`
                mutually exclusive with ``o_server``
            model_name: Name of the managed model in the ordered server,
                only needed if ``server`` needs such a identifier. The default
                ordered server does not require this.
            o_server: Custom ordered server accessor.
        """
        self.server_name = server_name
        self.group = group
        self.model_name = model_name
        # actual running server started by OrderedServerSimpleStarter
        self._o_server_impl = None
        if o_server is None:
            self._o_server_impl = OrderedServerSimpleImpl(
                server_name + "_o_server", group
            )
            self.o_server = group.get_paired(server_name + "_o_server").to_here()
        else:  # pragma: no cover
            self.o_server = o_server
        # pair an accessor to group
        self.group.pair(
            server_name, PushPullModelServer(self.model_name, self.o_server)
        )


class ReduceType(enum.Enum):
    REDUCE_PRIMARY = 0
    REDUCE_SECONDARY = 1


class PushPullGradServer:
    def __init__(
        self,
        server_name: str,
        group: RpcGroup,
        model_name: str,
        secondary_reducers: List[str],
        o_server: OrderedServerBase,
    ):
        self.group = group
        self.model_name = model_name
        self.o_server = o_server
        self.secondary_services = [
            server_name + "/" + m + "/_push_service" for m in secondary_reducers
        ]

    def push(self, model: nn.Module):
        """
        Push the gradients of your model, then pull the newest parameters.
         Its gradients will be cleared.

        Args:
            model: Model to push.

        Returns:
            True if push succeeded, else False.
        """
        # extract gradients from the model
        grad_dict = {}
        for k, v in model.named_parameters():
            if not hasattr(v, "grad") or not t.is_tensor(v.grad):  # pragma: no cover
                raise RuntimeError(f"Parameter {k} doesn't have gradient to push!")
            grad_dict[k] = deepcopy(v.grad).to("cpu")
        self.group.registered_sync(
            choice(self.secondary_services),
            args=(grad_dict, ReduceType.REDUCE_SECONDARY),
        )
        return self.pull(model)

    def pull(self, model: nn.Module):
        """
        Pull the newest model. Its gradients will be cleared.

        Args:
            model: Model to push.

        Returns:
            True if pull succeeded, else False.
        """
        model.zero_grad()
        params = self.o_server.pull(self.model_name)
        if params is not None:
            # params could be None if the master reducer has't performed
            # a single reduction operation yet
            prep_load_state_dict(model, params[0])
            return True
        else:
            return False


class PushPullGradServerImpl:
    """
    A simple parameter server, which synchronize model parameters
    by pushing gradients and pulling back new parameters, no strict
    order is guaranteed.

    Warning:
        ``DistributedDataParallel`` is not supported. since we cannot
        load state dictionary after creation.
    """

    REDUCE_MASTER = 0
    REDUCE_SLAVE = 1

    def __init__(
        self,
        server_name: str,
        group: RpcGroup,
        model_name: str = "model",
        primary_reducer: str = None,
        secondary_reducers: List[str] = None,
        o_server: OrderedServerBase = None,
        reduce_method: str = "sum",
        reduce_device: Union[t.device, str] = "cpu",
        reduce_batch_size: int = 4,
        max_queue_size: int = 64,
    ):
        """
        Note:
            You should initialize ``PushPullGradServer`` on all members of
            ``secondary_reducers``, and ``primary_reducer``. Both of them
            should be members of the ``group``.

        Note:
            Internally the primary reducer will push updated versions
            to the ordered server.

        Hint:
            Reduction is performed in a tree fashion:

            1. In the first step, clients will push new gradients to a
               random secondary reducer, and the secondary reducer will perform
               the first reduction pass, then secondary reducers will push
               their results to the primary reducer.
            2. In the second step, the primary reducer will reduce results
               from the secondary reducer to get the final reduced gradient
               dictionary (has the same structure as state_dict), and assign
               gradients to its **managed model**, and perform the
               optimization.
            3. In the final step, the primary reducer will push the final
               model to the model server group, then clients can pull the
               newest model.

        Args:
            server_name: Name of this server, used to registered
                the server as a paired class of ``group``.
            group: Server group.
            model_name: Name of the managed model in the ordered server,
                only needed if ``server`` needs such a identifier. The default
                ordered server does not require this.
            primary_reducer: Name of the process serving as the primary reducer,
                which collects reduced gradients from secondary reducers and
                perform the final reduction.
            secondary_reducers: Name of the process serving as secondary
                reducers.
            o_server: Custom ordered server accessor. By default, the ordered
                server is a :class:`.OrderedServerSimple` hosted on the primary
                reducer.
            reduce_method: "mean" or "sum"
            reduce_device: Device to perform reduction, by default it is "cpu".
            reduce_batch_size: Size of a single reduction batch, server will
                wait until the number of requests in the reduction queue have
                reached this size.
            max_queue_size: Maximum reduction request queue size.
        """
        self.server_name = server_name
        self.group = group
        self.model_name = model_name

        if primary_reducer is None:
            primary_reducer = group.get_group_members()[0]
        assert group.is_member(primary_reducer)
        assert group.is_member()

        # actual running server started by OrderedServerSimpleStarter
        self._o_server_impl = None
        self.o_server = None
        if o_server is None:
            if group.get_cur_name() == primary_reducer:
                self._o_server_impl = OrderedServerSimpleImpl(
                    server_name + "_o_server", group
                )
            self.o_server = OrderedServerSimple(server_name + "_o_server", group)
        else:  # pragma: no cover
            self.o_server = o_server

        if secondary_reducers is None:
            secondary_reducers = group.get_group_members()

        self.primary_reducer = primary_reducer
        self.primary_service = server_name + "/" + primary_reducer + "/_push_service"
        self.secondary_reducers = secondary_reducers
        self.secondary_services = [
            server_name + "/" + m + "/_push_service" for m in secondary_reducers
        ]
        # register secondary reducer service
        self.group.register(
            server_name + "/" + group.get_cur_name() + "/_push_service",
            self._push_service,
        )

        # pair an accessor to group
        if self.group.get_cur_name() == self.primary_reducer:
            self.group.pair(
                self.server_name,
                PushPullGradServer(
                    self.server_name,
                    self.group,
                    self.model_name,
                    self.secondary_reducers,
                    self.o_server,
                ),
            )

        # prepare to start the reduction sub-thread
        assert reduce_method in ("mean", "sum")
        assert max_queue_size > 1
        assert reduce_batch_size > 1
        assert max_queue_size > reduce_batch_size
        self.started = False
        self.reduce_method = reduce_method
        self.reduce_batch_size = reduce_batch_size
        self.reduce_device = reduce_device
        self.max_queue_size = max_queue_size
        self.model = None  # type: Union[nn.Module, None]
        self.optimizer = None
        self.lr_scheduler = None
        # do not set max_queue_size here, will raise queue.Full
        self.master_queue = Queue()
        self.secondary_queue = Queue()
        self.work_event = Event()
        self.stop_event = Event()
        self.reduce_task = Thread(target=self._task_reduce_grad)
        self.reduce_task.daemon = True

    def start(self):
        if not self.started:
            self.reduce_task.start()
            self.started = True

    def stop(self):
        if self.started:
            self.stop_event.set()
            self.reduce_task.join()
            self.stop_event.clear()

    def watch(self):
        self.reduce_task.watch()

    def manage_model(self, model: nn.Module, optimizer: Any, lr_scheduler: Any = None):
        """
        Let the main reducer manage your model. Must be called before start.

        Warning:
            Make sure that the managed model is different from the model
            you use in your algorithms such as A3C!

        Args:
            model: Model to manage.
            optimizer: Optimizer of your model. you should initialize it first:
            >>> optimizer = Adam(model.parameters(), lr=1e-3)

            lr_scheduler: learning rate scheduler, you should initialize it
                first:
            >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

        Raises:
            ``RuntimeError`` if current rpc role is not the main reducer.
        """
        if self.group.get_cur_name() == self.primary_reducer:
            self.model = model
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
            self.model.pp_version = 0
        else:  # pragma: no cover
            raise RuntimeError(
                "Current worker is not the reduce master, and"
                "cannot manage the model."
            )

    def _push_service(self, grad_dict, level):  # pragma: no cover
        # Append reduce requests to queue.
        if level == ReduceType.REDUCE_SECONDARY:
            self.secondary_queue.put_nowait(grad_dict)
            self.work_event.set()
            self.work_event.clear()
        elif level == ReduceType.REDUCE_PRIMARY:
            self.master_queue.put_nowait(grad_dict)
            self.work_event.set()
            self.work_event.clear()
        else:  # pragma: no cover
            raise ValueError(f"Unknown push level: {level}")

    def _task_reduce_grad(self):
        while True:
            # Wait until one queue has reached target batch size
            while (
                self.master_queue.qsize() < self.reduce_batch_size
                and self.secondary_queue.qsize() < self.reduce_batch_size
            ):
                self.work_event.wait(timeout=1e-1)
                if self.stop_event.is_set():
                    return
            # discard oldest messages
            while self.master_queue.qsize() > self.max_queue_size:
                self.master_queue.get()
            while self.secondary_queue.qsize() > self.max_queue_size:
                self.secondary_queue.get()

            if self.master_queue.qsize() >= self.reduce_batch_size:
                # Perform reduction on the master reduction queue
                # Only the master reducer will execute this branch
                grad_dict = self._reduce_batch(
                    self.master_queue,
                    self.reduce_batch_size,
                    self.reduce_method,
                    self.reduce_device,
                )
                # Assign gradients to the managed model and
                # perform optimization.
                if self.model is not None and self.optimizer is not None:
                    self.optimizer.zero_grad()
                    with t.no_grad():
                        for k, v in self.model.named_parameters():
                            v.grad = grad_dict[k].to(v.device)
                    self.optimizer.step()
                    self.o_server.push(
                        self.model_name,
                        self.model.to("cpu").state_dict(),
                        self.model.pp_version + 1,
                        self.model.pp_version,
                    )
                    self.model.pp_version += 1

            if self.secondary_queue.qsize() >= self.reduce_batch_size:
                # Perform reduction on the secondary reduction queue
                # All processes(including master) in the reduction
                # group will execute this branch.
                grad_dict = self._reduce_batch(
                    self.secondary_queue,
                    self.reduce_batch_size,
                    self.reduce_method,
                    self.reduce_device,
                )
                # Push reduced results to the master queue.
                self.group.registered_sync(
                    self.primary_service, args=(grad_dict, ReduceType.REDUCE_PRIMARY)
                )

    @staticmethod
    def _reduce_batch(queue, batch_size, reduce_method, reduce_device):
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
        for k, v in grad_dict.items():
            # Stack parameter tensors in dim 0 and reduce.
            if reduce_method == "sum":
                grad_dict[k] = t.sum(t.stack(v, dim=0), dim=0, keepdim=False)
            elif reduce_method == "mean":
                grad_dict[k] = t.mean(t.stack(v, dim=0), dim=0, keepdim=False)
            else:  # pragma: no cover
                raise RuntimeError("Unknown reduce method.")
        return grad_dict
