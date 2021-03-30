from machin.auto.config import (
    generate_training_config,
    generate_algorithm_config,
    init_algorithm_from_config,
)
from machin.auto.env.openai_gym import (
    RLGymDiscActDataset,
    RLGymContActDataset,
    generate_gym_env_config,
    gym_env_dataset_creator,
    launch_gym,
)
from test.util_run_multi import *
from test.util_fixtures import *
from pytorch_lightning.callbacks import Callback
from torch.distributions import Categorical, Normal
from machin.utils.logging import default_logger
import gym
import pytest
import torch as t
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


class A2CActorCont(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(A2CActorCont, self).__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.mu_head = nn.Linear(16, action_dim)
        self.sigma_head = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        mu = 2.0 * t.tanh(self.mu_head(a))
        sigma = F.softplus(self.sigma_head(a))
        dist = Normal(mu, sigma)
        action = action if action is not None else dist.sample()
        action_entropy = dist.entropy()
        action = action.clamp(-self.action_range, self.action_range)
        action_log_prob = dist.log_prob(action)
        return action, action_log_prob, action_entropy


class A2CActorDisc(nn.Module):
    def __init__(self, state_dim, action_num):
        super(A2CActorDisc, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class A2CCritic(nn.Module):
    def __init__(self, state_dim):
        super(A2CCritic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


class DDPGActorCont(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(DDPGActorCont, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a


class DDPGActorDisc(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGActorDisc, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.softmax(self.fc3(a), dim=1)
        return a


class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TestRLGymDiscActDataset:
    @staticmethod
    def assert_valid_disc_output(result):
        assert len(result.observations) > 0
        assert len(result.logs) > 0
        assert t.is_tensor(result.observations[0]["state"]["state"])
        assert t.is_tensor(result.observations[0]["action"]["action"])
        assert t.is_tensor(result.observations[0]["next_state"]["state"])
        assert isinstance(result.observations[0]["reward"], float)
        assert isinstance(result.observations[0]["terminal"], bool)
        log_keys = set([k for log_dict in result.logs for k in log_dict])
        assert log_keys.issuperset({"video", "total_reward"})

    # Only test single node, most representative algorithms
    def test_A2C(self):
        config = generate_algorithm_config("A2C", {})
        config["frame_config"]["models"] = ["A2CActorDisc", "A2CCritic"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": 4, "action_num": 2},
            {"state_dim": 4},
        ]
        a2c = init_algorithm_from_config(config)

        env = gym.make("CartPole-v0")
        dataset = RLGymDiscActDataset(a2c, env, render_every_episode=1)
        self.assert_valid_disc_output(next(dataset))

    def test_DQN(self):
        config = generate_algorithm_config("DQN", {})
        config["frame_config"]["models"] = ["QNet", "QNet"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": 4, "action_num": 2},
            {"state_dim": 4, "action_num": 2},
        ]
        dqn = init_algorithm_from_config(config)

        env = gym.make("CartPole-v0")
        dataset = RLGymDiscActDataset(dqn, env, render_every_episode=1)
        self.assert_valid_disc_output(next(dataset))

    def test_DDPG(self):
        config = generate_algorithm_config("DDPG", {})
        config["frame_config"]["models"] = [
            "DDPGActorDisc",
            "DDPGActorDisc",
            "DDPGCritic",
            "DDPGCritic",
        ]
        config["frame_config"]["model_kwargs"] = [{"state_dim": 4, "action_dim": 2}] * 4
        ddpg = init_algorithm_from_config(config)

        env = gym.make("CartPole-v0")
        dataset = RLGymDiscActDataset(ddpg, env, render_every_episode=1)
        self.assert_valid_disc_output(next(dataset))


class TestRLGymContActDataset:
    @staticmethod
    def assert_valid_cont_output(result):
        assert len(result.observations) > 0
        assert len(result.logs) > 0
        assert t.is_tensor(result.observations[0]["state"]["state"])
        assert t.is_tensor(result.observations[0]["action"]["action"])
        assert t.is_tensor(result.observations[0]["next_state"]["state"])
        assert isinstance(result.observations[0]["reward"], float)
        assert isinstance(result.observations[0]["terminal"], bool)
        log_keys = set([k for log_dict in result.logs for k in log_dict])
        assert log_keys.issuperset({"video", "total_reward"})

    # Only test single node, most representative algorithms
    def test_A2C(self):
        config = generate_algorithm_config("A2C", {})
        config["frame_config"]["models"] = ["A2CActorCont", "A2CCritic"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": 3, "action_dim": 1, "action_range": 2},
            {"state_dim": 3},
        ]
        a2c = init_algorithm_from_config(config)

        env = gym.make("Pendulum-v0")
        dataset = RLGymContActDataset(a2c, env, render_every_episode=1)
        self.assert_valid_cont_output(next(dataset))

    def test_DDPG(self):
        config = generate_algorithm_config("DDPG", {})
        config["frame_config"]["models"] = [
            "DDPGActorCont",
            "DDPGActorCont",
            "DDPGCritic",
            "DDPGCritic",
        ]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": 3, "action_dim": 1, "action_range": 2}
        ] * 2 + [{"state_dim": 3, "action_dim": 1}] * 2
        ddpg = init_algorithm_from_config(config)

        env = gym.make("Pendulum-v0")
        dataset = RLGymContActDataset(ddpg, env, render_every_episode=1)
        self.assert_valid_cont_output(next(dataset))


def test_gym_env_dataset_creator():
    # Discrete action environment
    config = generate_gym_env_config("CartPole-v0", {})
    config = generate_algorithm_config("DDPG", config)
    config["frame_config"]["models"] = [
        "DDPGActorCont",
        "DDPGActorCont",
        "DDPGCritic",
        "DDPGCritic",
    ]
    config["frame_config"]["model_kwargs"] = [
        {"state_dim": 3, "action_dim": 1, "action_range": 2}
    ] * 2 + [{"state_dim": 3, "action_dim": 1}] * 2
    ddpg = init_algorithm_from_config(config)

    assert isinstance(
        gym_env_dataset_creator(ddpg, config["train_env_config"]), RLGymDiscActDataset
    )
    assert isinstance(
        gym_env_dataset_creator(ddpg, config["test_env_config"]), RLGymDiscActDataset
    )

    # Continuous action environment
    config = generate_gym_env_config("Pendulum-v0", {})
    assert isinstance(
        gym_env_dataset_creator(ddpg, config["train_env_config"]), RLGymContActDataset
    )
    assert isinstance(
        gym_env_dataset_creator(ddpg, config["test_env_config"]), RLGymContActDataset
    )

    # Unsupported environment,
    # like algorithmic, which uses a tuple action space
    # or robotics, which uses the goal action space
    config = generate_gym_env_config("Copy-v0", {})
    with pytest.raises(ValueError, match="not supported"):
        gym_env_dataset_creator(ddpg, config["train_env_config"])

    with pytest.raises(ValueError, match="not supported"):
        gym_env_dataset_creator(ddpg, config["test_env_config"])


class InspectCallback(Callback):
    def __init__(self):
        self.max_total_reward = 0

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, _batch_idx, _dataloader_idx
    ) -> None:
        for log in batch[0].logs:
            if "total_reward" in log:
                self.max_total_reward = max(log["total_reward"], self.max_total_reward)
                default_logger.info(
                    f"Current max total reward={self.max_total_reward:.2f}."
                )
                if self.max_total_reward >= 190:
                    trainer.should_stop = True
                return
        default_logger.error("Missing total reward in logs.")


class TestLaunchGym:
    def test_dqn(self, tmpdir):
        config = generate_gym_env_config("CartPole-v0", {})
        config = generate_training_config(
            trials_dir=str(tmpdir.make_numbered_dir()), config=config
        )
        config = generate_algorithm_config("DQN", config)
        config["early_stopping_patience"] = 10
        config["frame_config"]["models"] = ["QNet", "QNet"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": 4, "action_num": 2},
            {"state_dim": 4, "action_num": 2},
        ]
        cb = InspectCallback()
        launch_gym(config, pl_callbacks=[cb])
        assert cb.max_total_reward >= 150

    @staticmethod
    @run_multi(expected_results=[True, True, True], pass_through="tmpdir", timeout=1800)
    @WorldTestBase.setup_world
    def test_dqn_apex(_, tmpdir):
        config = generate_gym_env_config("CartPole-v0", {})
        config = generate_training_config(
            trials_dir=str(tmpdir.make_numbered_dir()), config=config
        )
        config = generate_algorithm_config("DQNApex", config)
        config["early_stopping_patience"] = 10
        config["frame_config"]["models"] = ["QNet", "QNet"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": 4, "action_num": 2},
            {"state_dim": 4, "action_num": 2},
        ]
        cb = InspectCallback()
        launch_gym(config, pl_callbacks=[cb])
        assert cb.max_total_reward >= 150
