from machin.frame.algorithms import A2C, DDPG, DQN
from machin.auto.env.openai_gym import RLGymDiscActDataset, RLGymContActDataset
from torch.distributions import Categorical, Normal
import gym
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
        config = A2C.generate_config({})
        config["frame_config"]["models"] = ["A2CActorDisc", "A2CCritic"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": 4, "action_num": 2},
            {"state_dim": 4},
        ]
        a2c = A2C.init_from_config(config)

        env = gym.make("CartPole-v0")
        dataset = RLGymDiscActDataset(a2c, env, render_every_episode=1)
        self.assert_valid_disc_output(next(dataset))

    def test_DQN(self):
        config = DQN.generate_config({})
        config["frame_config"]["models"] = ["QNet", "QNet"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": 4, "action_num": 2},
            {"state_dim": 4, "action_num": 2},
        ]
        dqn = DQN.init_from_config(config)

        env = gym.make("CartPole-v0")
        dataset = RLGymDiscActDataset(dqn, env)
        self.assert_valid_disc_output(next(dataset))

    def test_DDPG(self):
        config = DDPG.generate_config({})
        config["frame_config"]["models"] = [
            "DDPGActorDisc",
            "DDPGActorDisc",
            "DDPGCritic",
            "DDPGCritic",
        ]
        config["frame_config"]["model_kwargs"] = [{"state_dim": 4, "action_dim": 2}] * 4
        ddpg = DDPG.init_from_config(config)

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
        config = A2C.generate_config({})
        config["frame_config"]["models"] = ["A2CActorCont", "A2CCritic"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": 3, "action_dim": 1, "action_range": 2},
            {"state_dim": 3},
        ]
        a2c = A2C.init_from_config(config)

        env = gym.make("Pendulum-v0")
        dataset = RLGymContActDataset(a2c, env, render_every_episode=1)
        self.assert_valid_cont_output(next(dataset))

    def test_DDPG(self):
        config = DDPG.generate_config({})
        config["frame_config"]["models"] = [
            "DDPGActorCont",
            "DDPGActorCont",
            "DDPGCritic",
            "DDPGCritic",
        ]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": 3, "action_dim": 1, "action_range": 2}
        ] * 2 + [{"state_dim": 3, "action_dim": 1}] * 2
        ddpg = DDPG.init_from_config(config)

        env = gym.make("Pendulum-v0")
        dataset = RLGymContActDataset(ddpg, env, render_every_episode=1)
        self.assert_valid_cont_output(next(dataset))
