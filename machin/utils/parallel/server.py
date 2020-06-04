from typing import Any, Union
from collections import OrderedDict
from threading import Lock, Condition
from random import choice
from copy import deepcopy
from queue import Queue
from torch.nn.parallel import DistributedDataParallel

import torch as t
import torch.nn as nn

from .distributed import RpcGroup
from machin.utils.prep import prep_load_state_dict


class SimpleOrderedServer:
    def __init__(self, group: RpcGroup, master=0, log_depth=1, **__):
        self.group = group
        self.group.rpc_register_paired(self.__class__, self)
        self.storage = {}
        self.lock = Lock()
        self.log_depth = log_depth
        self.master = self.group.get_group_members()[master]
        self.slaves = [member
                       for member in self.group.get_group_members()
                       if member != self.master]

        assert log_depth > 0 and isinstance(log_depth, int)

    def push(self, key, value, version, prev_version):
        return self.group.rpc_paired_class_sync(self.master,
                                                self._reply_push,
                                                self.__class__,
                                                args=(key, value, version,
                                                      prev_version))

    def pull(self, key, version=None):
        to = choice(self.group.get_group_members())
        return self.group.rpc_paired_class_sync(to,
                                                self._reply_pull,
                                                self.__class__,
                                                args=(key, version))[1]

    def latest(self, key):
        to = choice(self.group.get_group_members())
        return self.group.rpc_paired_class_sync(to,
                                                self._reply_pull,
                                                self.__class__,
                                                args=(key, None))[1]

    def _reply_push(self, key, value, version, prev_version):
        success = False
        self.lock.acquire()
        if key in self.storage:
            ref = self.storage[key]
            if next(reversed(ref)) == prev_version:
                ref[version] = value
                success = True
            if len(ref) > self.log_depth + 1:
                ref.pop(0)
        else:
            ref = self.storage[key] = OrderedDict()
            ref[version] = value
            success = True

        future = []
        for slave in self.slaves:
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
            if version is not None and version in ref:
                result = (version, ref[version])
            elif version is None:
                result = (next(reversed(ref)), ref[-1])
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


class PushPullModelServer:
    def __init__(self, group, master=0, server=None):
        self.server = (SimpleOrderedServer(group, master=master)
                       if server is None
                       else server)

    def push(self, model, name):
        if not hasattr(model, "pp_version"):
            model.pp_version = 0
        model = deepcopy(model).to("cpu")

        need_sync = self.server.group.size() > 1 and \
                    not isinstance(model, DistributedDataParallel)

        if not self.server.push(
                name, model.to("cpu").state_dict(),
                version=model.pp_version + 1, prev_version=model.pp_version
        ) and need_sync:
            version, st_dict = self.server.pull(name)
            model.load_state_dict(st_dict)
            model.pp_version = version

    def pull(self, model, name):
        version, st_dict = self.server.pull(name)
        if not hasattr(model, "pp_version") or model.pp_version < version:
            prep_load_state_dict(model, st_dict)
            model.pp_version = version


class PushPullGradServer:
    REDUCE_MASTER = 0
    REDUCE_SLAVE = 1

    def __init__(self, group,
                 reduce_master=0,
                 model_master=0,
                 server=None,
                 reduce_device="cpu",
                 reduce_batch_size=64,
                 max_queue_size=1024):
        self.server = (SimpleOrderedServer(group, master=model_master)
                       if server is None
                       else server)
        self.group.rpc_register_paired(self.__class__, self)
        self.group = group
        self.reduce_master = self.group.get_group_members()[reduce_master]
        self.reduce_batch_size = reduce_batch_size
        self.reduce_device = reduce_device
        self.master_queue = Queue(maxsize=max_queue_size)
        self.slave_queue = Queue(maxsize=max_queue_size)
        self.work_cond = Condition(Lock())
        self.model = None  # type: Union[nn.Module, None]
        self.optimizer = None

    def push(self, model: nn.Module, *_, **__):
        # extract gradients from the model
        gradients = {}
        for k, v in model.parameters():
            gradients[k] = v

    def pull(self, model: nn.Module, *_, **__):
        params = self.server.pull("_push_pull_grad_managed_model")
        prep_load_state_dict(model, params)

    def manage_model(self, model: nn.Module, optimizer: Any):
        if self.group.get_cur_role() == self.reduce_master:
            self.model = model
            self.optimizer = optimizer
        else:
            raise RuntimeError("Current worker is not reduce master, and"
                               "cannot manage the model.")

    def _push_reply(self, grad_dict, level):
        if level == self.REDUCE_SLAVE:
            self.slave_queue.put_nowait(grad_dict)
            self.work_cond.notify_all()
        elif level == self.REDUCE_MASTER:
            self.master_queue.put_nowait(grad_dict)
            self.work_cond.notify_all()

    def _task_reduce_grad(self):
        while True:
            self.work_cond.wait_for(
                lambda: (self.master_queue.qsize() > self.reduce_batch_size or
                         self.slave_queue.qsize() > self.reduce_batch_size)
            )
            if self.master_queue.qsize() > self.reduce_batch_size:
                grad = self._reduce_batch(self.master_queue,
                                          self.reduce_batch_size,
                                          self.reduce_device)
                if self.model is not None and self.optimizer is not None:
                    self.optimizer.zero_grad()
                    with t.no_grad():
                        for k, v in self.model.parameters():
                            v.grad = grad[k].to(v.device)
                    self.optimizer.step()
                self.server.push("_push_pull_grad_managed_model",
                                 self.model.to("cpu").parameters(), 0, 0)

            elif self.slave_queue.qsize() > self.reduce_batch_size:
                grad = self._reduce_batch(self.master_queue,
                                          self.reduce_batch_size,
                                          self.reduce_device)
                self.group.rpc_paired_class_async(
                    self.reduce_master,
                    self._push_reply,
                    self.__class__,
                    args=(grad, self.REDUCE_MASTER)
                )

    @staticmethod
    def _reduce_batch(queue, batch_size, reduce_device):
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
            grad_dict[k] = t.mean(t.stack(v), dim=0, keepdim=True)
        return grad_dict
