from machin.parallel.server import PushPullGradServerImpl, PushPullModelServerImpl
from test.util_run_multi import *
import random
import torch as t
import torch.nn as nn


def _log(rank, msg):
    default_logger.info(f"Client {rank}: {msg}")


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
            self.fc3.weight.grad.item(),
        )


class Optimizer(object):
    def __init__(self, param):
        self.params = param

    def zero_grad(self):
        pass

    def step(self):
        with t.no_grad():
            for p in self.params:
                default_logger.critical(p.grad)
                p -= p.grad


class TestPushPullModelServer(WorldTestBase):
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_push_pull(rank):
        world = get_world()
        if rank == 0:
            group = world.create_rpc_group("group", ["0", "1"])
            _server = PushPullModelServerImpl("server", group)
            group.barrier()
            group.barrier()
        elif rank == 1:
            group = world.create_rpc_group("group", ["0", "1"])
            group.barrier()
            server = group.get_paired("server").to_here()
            model = Model()
            pull_model = Model()
            for i in range(10):
                with t.no_grad():
                    model.fc1.weight += 1
                    model.fc2.weight += 1
                    model.fc3.weight += 1
                if server.push(model):
                    _log(rank, f"push {i} success")
                else:
                    _log(rank, f"push {i} failed")
                model.pp_version = i + 1
                _log(
                    rank,
                    "model: {}, {}, {}".format(
                        model.fc1.weight.item(),
                        model.fc2.weight.item(),
                        model.fc3.weight.item(),
                    ),
                )
                sleep(random.random() * 0.2)
            server.pull(pull_model)
            assert pull_model.fc1.weight.item() == 11
            assert pull_model.fc2.weight.item() == 12
            assert pull_model.fc3.weight.item() == 13
            group.barrier()
        return True


class TestPushPullGradServer(WorldTestBase):
    @staticmethod
    @pytest.mark.parametrize(
        "reduce_method,new_weight", [("mean", (-5, -1, 1)), ("sum", (-23, -10, -5))]
    )
    @run_multi(
        expected_results=[True, True, True],
        pass_through=["reduce_method", "new_weight"],
    )
    @WorldTestBase.setup_world
    def test_push_pull(rank, reduce_method, new_weight):
        world = get_world()
        if rank == 0:
            # only one reduce slave, so result is controllable
            group = world.create_rpc_group("group", ["0", "1", "2"])
            server = PushPullGradServerImpl(
                "server",
                group,
                primary_reducer="0",
                secondary_reducers=["0"],
                reduce_method=reduce_method,
                reduce_batch_size=2,
            )
            group.barrier()
            model = Model()
            server.manage_model(model, Optimizer(model.parameters()))
            begin = time()
            server.start()
            while time() - begin < 5:
                server.watch()
            server.stop()
            group.barrier()
        else:
            group = world.create_rpc_group("group", ["0", "1", "2"])
            group.barrier()
            server = group.get_paired("server").to_here()
            model = Model()

            # "0" will be wake up twice as a slave reducer, and once
            # as the master reducer.
            for i in range(2):
                model.zero_grad()
                loss = model(t.ones([1, 1]))
                loss.backward()
                server.push(model)
                _log(rank, f"iter {i}, model: {model}")
                sleep(random.random() * 0.2)
            sleep(3)
            server.pull(model)
            _log(rank, f"reduced model: {model}")
            # reduce_method = "mean":
            # fc1: weight(1) - 6 = -5
            # fc2: weight(2) - 3 = -1
            # fc3: weight(3) - 2 = 1
            # reduce_method = "sum":
            # fc1: weight(1) - 4 * 6 = -23
            # fc2: weight(2) - 4 * 3 = -10
            # fc3: weight(3) - 4 * 2 = -5
            assert model.fc1.weight.item() == new_weight[0]
            assert model.fc2.weight.item() == new_weight[1]
            assert model.fc3.weight.item() == new_weight[2]
            group.barrier()
        return True
