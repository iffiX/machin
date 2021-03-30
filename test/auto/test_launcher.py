from machin.frame.algorithms import DQN
from machin.auto.launcher import Launcher
from machin.auto.dataset import RLDataset, DatasetResult
from unittest import mock
import pytest
import torch as t
import torch.nn as nn
import pytorch_lightning as pl


class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


class FakeDataset(RLDataset):
    def __next__(self):
        dr = DatasetResult()
        dr.add_observation(
            {
                "state": {"state": t.ones([1, 4])},
                "action": {"action": t.ones([1, 1], dtype=t.long)},
                "next_state": {"state": t.ones([1, 4])},
                "terminal": False,
                "reward": 1,
            }
        )
        dr.add_log({"test_log": (1, set_test_var)})
        return dr


test_var = None


def set_test_var(_launcher, log_key, log_val):
    global test_var
    test_var = {log_key: log_val}


def generate_config():
    config = DQN.generate_config({})
    config["frame_config"]["models"] = ["QNet", "QNet"]
    config["frame_config"]["model_kwargs"] = [{"state_dim": 4, "action_num": 2}] * 2
    config["train_env_config"] = {}
    config["test_env_config"] = {}
    return config


def fake_env_dataset_creator(_frame, _env_config):
    return FakeDataset()


class TestLauncher:
    @pytest.fixture(scope="function")
    def launcher(self):
        l = Launcher(generate_config(), fake_env_dataset_creator)
        l.trainer = mock.MagicMock()
        l.trainer.accelerator_connector.use_ddp = False
        l.trainer.train_loop.automatic_optimization = False
        l.optimizers = lambda: l.frame.optimizers
        return l

    @pytest.fixture(scope="function")
    def real_launcher(self):
        l = Launcher(generate_config(), fake_env_dataset_creator)
        return l

    def test_on_train_start(self, launcher):
        launcher.on_train_start()
        assert launcher.frame.backward_function == launcher.manual_backward
        assert launcher.frame.optimizers == launcher.optimizers()

    def test_on_test_start(self, launcher):
        launcher.on_test_start()
        assert launcher.frame.backward_function == launcher.manual_backward
        assert launcher.frame.optimizers == launcher.optimizers()

    def test_train_dataloader(self, launcher):
        dl = launcher.train_dataloader()
        dr = next(iter(dl))[0]
        assert isinstance(dr, DatasetResult)
        assert len(dr.observations) > 0 and "state" in dr.observations[0]

    def test_test_dataloader(self, launcher):
        dl = launcher.test_dataloader()
        dr = next(iter(dl))[0]
        assert isinstance(dr, DatasetResult)
        assert len(dr.observations) > 0 and "state" in dr.observations[0]

    def test_training_step(self, launcher):
        global test_var
        dl = launcher.train_dataloader()
        dr = next(iter(dl))
        launcher.training_step(dr, 0)
        assert test_var == {"test_log": 1}
        test_var = None

    def test_test_step(self, launcher):
        global test_var
        dl = launcher.train_dataloader()
        dr = next(iter(dl))
        launcher.training_step(dr, 0)
        assert test_var == {"test_log": 1}
        test_var = None

    def test_configure_optimizers(self, launcher):
        optims, schs = launcher.configure_optimizers()
        # DQN has 1 optimizer
        assert len(optims) == 1
        assert isinstance(optims[0], t.optim.Optimizer)
        assert len(schs) == 0

    def test_real(self, real_launcher):
        trainer = pl.Trainer(
            gpus=0, limit_train_batches=1, max_steps=1, automatic_optimization=False,
        )
        trainer.fit(real_launcher)
