from machin.parallel.server import (
    PushPullGradServer,
    PushPullModelServer
)
from ..util_run_multi import *
import random
import torch as t
import torch.nn as nn
from torch.optim import Adam


def _log(rank, msg):
    default_logger.info("Client {}: {}".format(rank, msg))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.fc2 = nn.Linear(1, 1, bias=False)
        self.fc3 = nn.Linear(1, 1, bias=False)

        with t.no_grad():
            self.fc1.weight.fill_(1)
            self.fc2.weight.fill_(2)
            self.fc3.weight.fill_(3)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))

    def __repr__(self):
        return "Model(param=({}, {}, {}), grad=({}, {}, {}))".format(
            self.fc1.weight.item(),
            self.fc2.weight.item(),
            self.fc3.weight.item(),
            self.fc1.weight.grad.item(),
            self.fc2.weight.grad.item(),
            self.fc3.weight.grad.item()
        )


class TestPushPullModelServer(WorldTestBase):
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_push_pull(rank):
        world = get_world()
        if rank == 0:
            group = world.create_rpc_group("group", ["0"])
            _server = PushPullModelServer("model", group)
            sleep(3)
        else:
            group = world.get_rpc_group("group", "0")
            server = PushPullModelServer("model", group)
            model = Model()
            pull_model = Model()
            for i in range(10):
                with t.no_grad():
                    model.fc1.weight += 1
                    model.fc2.weight += 1
                    model.fc3.weight += 1
                if server.push(model):
                    _log(rank, "push {} success".format(i))
                else:
                    _log(rank, "push {} failed".format(i))
                model.pp_version = i + 1
                _log(rank, "model: {}, {}, {}".format(
                    model.fc1.weight.item(),
                    model.fc2.weight.item(),
                    model.fc3.weight.item()
                ))
                sleep(random.random() * 0.2)
            server.pull(pull_model)
            assert pull_model.fc1.weight.item() == 11
            assert pull_model.fc2.weight.item() == 12
            assert pull_model.fc3.weight.item() == 13
        return True


class TestPushPullGradServer(WorldTestBase):
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_push_pull(rank):
        world = get_world()
        if rank == 0:
            # only one reduce slave, so result is controllable
            group = world.create_rpc_group("group", ["0"])
            server = PushPullGradServer("model", group, reduce_batch_size=2)
            model = Model()
            server.manage_model(model, Adam(model.parameters(), lr=1))
            begin = time()
            server.start()
            while time() - begin < 5:
                server.watch()
            server.stop()
        else:
            group = world.get_rpc_group("group", "0")
            server = PushPullGradServer("model", group, reduce_batch_size=2)
            model = Model()

            # "0" will be wake up twice as a slave reducer, and once
            # as the master reducer.
            for i in range(2):
                model.zero_grad()
                loss = model(t.ones([1, 1]))
                loss.backward()
                server.push(model)
                _log(rank, "iter {}, model: {}".format(i, model))
                sleep(random.random() * 0.2)
            sleep(3)
            server.pull(model)
            _log(rank, "reduced model: {}".format(model))
            assert model.fc1.weight.item() == 0
            assert model.fc2.weight.item() == 1
            assert model.fc3.weight.item() == 2
        return True
