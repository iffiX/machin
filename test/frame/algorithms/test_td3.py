from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.td3 import TD3
from machin.utils.learning_rate import gen_learning_rate_func
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
from torch.optim.lr_scheduler import LambdaLR

import pytest
import torch as t
import torch.nn as nn
import gym

from .utils import unwrap_time_limit, Smooth
from test.util_fixtures import *


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a


class ActorDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorDiscrete, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.softmax(self.fc3(a), dim=1)
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TestTD3(object):
    # configs and definitions
    @pytest.fixture(scope="class")
    def train_config(self):
        disable_view_window()
        c = Config()
        c.env_name = "Pendulum-v0"
        c.env = unwrap_time_limit(gym.make(c.env_name))
        c.observe_dim = 3
        c.action_dim = 1
        c.action_range = 2
        c.max_episodes = 1000
        c.max_steps = 200
        c.noise_param = (0, 0.2)
        c.noise_mode = "normal"
        c.noise_interval = 2
        c.replay_size = 100000
        c.solved_reward = -400
        c.solved_repeat = 5
        return c

    @pytest.fixture(scope="function")
    def td3(self, train_config, device, dtype):
        c = train_config
        actor = smw(
            Actor(c.observe_dim, c.action_dim, c.action_range).type(dtype).to(device),
            device,
            device,
        )
        actor_t = smw(
            Actor(c.observe_dim, c.action_dim, c.action_range).type(dtype).to(device),
            device,
            device,
        )
        critic = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2 = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        td3 = TD3(
            actor,
            actor_t,
            critic,
            critic_t,
            critic2,
            critic2_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        return td3

    @pytest.fixture(scope="function")
    def td3_vis(self, train_config, device, dtype, tmpdir):
        # not used for training, only used for testing apis
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        actor = smw(
            Actor(c.observe_dim, c.action_dim, c.action_range).type(dtype).to(device),
            device,
            device,
        )
        actor_t = smw(
            Actor(c.observe_dim, c.action_dim, c.action_range).type(dtype).to(device),
            device,
            device,
        )
        critic = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2 = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        td3 = TD3(
            actor,
            actor_t,
            critic,
            critic_t,
            critic2,
            critic2_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
            visualize=True,
            visualize_dir=str(tmp_dir),
        )
        return td3

    @pytest.fixture(scope="function")
    def td3_lr(self, train_config, device, dtype):
        # not used for training, only used for testing apis
        c = train_config
        actor = smw(
            ActorDiscrete(c.observe_dim, c.action_dim).type(dtype).to(device),
            device,
            device,
        )
        actor_t = smw(
            ActorDiscrete(c.observe_dim, c.action_dim).type(dtype).to(device),
            device,
            device,
        )
        critic = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2 = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)], logger=logger)
        with pytest.raises(TypeError, match="missing .+ positional argument"):
            _ = TD3(
                actor,
                actor_t,
                critic,
                critic_t,
                critic2,
                critic2_t,
                t.optim.Adam,
                nn.MSELoss(reduction="sum"),
                replay_device="cpu",
                replay_size=c.replay_size,
                lr_scheduler=LambdaLR,
            )
        td3 = TD3(
            actor,
            actor_t,
            critic,
            critic_t,
            critic2,
            critic2_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
            lr_scheduler=LambdaLR,
            lr_scheduler_args=((lr_func,), (lr_func,), (lr_func,)),
        )
        return td3

    @pytest.fixture(scope="function")
    def td3_train(self, train_config):
        c = train_config
        # cpu is faster for testing full training.
        actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range), "cpu", "cpu")
        actor_t = smw(Actor(c.observe_dim, c.action_dim, c.action_range), "cpu", "cpu")
        critic = smw(Critic(c.observe_dim, c.action_dim), "cpu", "cpu")
        critic_t = smw(Critic(c.observe_dim, c.action_dim), "cpu", "cpu")
        critic2 = smw(Critic(c.observe_dim, c.action_dim), "cpu", "cpu")
        critic2_t = smw(Critic(c.observe_dim, c.action_dim), "cpu", "cpu")
        td3 = TD3(
            actor,
            actor_t,
            critic,
            critic_t,
            critic2,
            critic2_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        return td3

    ########################################################################
    # Test for TD3 contiguous domain acting
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for TD3 discrete domain acting
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for TD3 criticizing
    ########################################################################
    def test__criticize(self, train_config, td3, dtype):
        c = train_config
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, c.action_dim], dtype=dtype)
        td3._criticize({"state": state}, {"action": action})
        td3._criticize({"state": state}, {"action": action}, use_target=True)
        td3._criticize2({"state": state}, {"action": action})
        td3._criticize2({"state": state}, {"action": action}, use_target=True)

    ########################################################################
    # Test for TD3 storage
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for TD3 update
    ########################################################################
    def test_update(self, train_config, td3_vis, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, c.action_dim], dtype=dtype)
        td3_vis.store_transition(
            {
                "state": {"state": old_state},
                "action": {"action": action},
                "next_state": {"state": state},
                "reward": 0,
                "terminal": False,
            }
        )
        td3_vis.update(
            update_value=True,
            update_policy=True,
            update_target=True,
            concatenate_samples=True,
        )
        td3_vis.update(
            update_value=False,
            update_policy=False,
            update_target=False,
            concatenate_samples=True,
        )

    ########################################################################
    # Test for TD3 save & load
    ########################################################################
    def test_save_load(self, train_config, td3, tmpdir):
        save_dir = tmpdir.make_numbered_dir()
        td3.save(
            model_dir=str(save_dir),
            network_map={
                "critic_target": "critic_t",
                "critic2_target": "critic2_t",
                "actor_target": "actor_t",
            },
            version=1000,
        )
        td3.load(
            model_dir=str(save_dir),
            network_map={
                "critic_target": "critic_t",
                "critic2_target": "critic2_t",
                "actor_target": "actor_t",
            },
            version=1000,
        )

    ########################################################################
    # Test for TD3 lr_scheduler
    ########################################################################
    def test_lr_scheduler(self, train_config, td3_lr):
        td3_lr.update_lr_scheduler()

    ########################################################################
    # Test for TD3 config & init
    ########################################################################
    def test_config_init(self, train_config):
        c = train_config
        config = TD3.generate_config({})
        config["frame_config"]["models"] = [
            "Actor",
            "Actor",
            "Critic",
            "Critic",
            "Critic",
            "Critic",
        ]
        config["frame_config"]["model_kwargs"] = [
            {
                "state_dim": c.observe_dim,
                "action_dim": c.action_dim,
                "action_range": c.action_range,
            }
        ] * 2 + [{"state_dim": c.observe_dim, "action_dim": c.action_dim}] * 4
        td3 = TD3.init_from_config(config)

        old_state = state = t.zeros([1, c.observe_dim], dtype=t.float32)
        action = t.zeros([1, c.action_dim], dtype=t.float32)
        td3.store_transition(
            {
                "state": {"state": old_state},
                "action": {"action": action},
                "next_state": {"state": state},
                "reward": 0,
                "terminal": False,
            }
        )
        td3.update()

    ########################################################################
    # Test for TD3 full training.
    ########################################################################
    def test_full_train(self, train_config, td3_train):
        c = train_config

        # begin training
        episode, step = Counter(), Counter()
        reward_fulfilled = Counter()
        smoother = Smooth()
        terminal = False

        env = c.env
        while episode < c.max_episodes:
            episode.count()

            # batch size = 1
            total_reward = 0
            state = t.tensor(env.reset(), dtype=t.float32)

            while not terminal and step <= c.max_steps:
                step.count()
                with t.no_grad():
                    old_state = state

                    # agent model inference
                    if episode.get() % c.noise_interval == 0:
                        action = td3_train.act_with_noise(
                            {"state": old_state.unsqueeze(0)},
                            noise_param=c.noise_param,
                            mode=c.noise_mode,
                        )
                    else:
                        action = td3_train.act({"state": old_state.unsqueeze(0)}).clamp(
                            -c.action_range, c.action_range
                        )

                    state, reward, terminal, _ = env.step(action.cpu().numpy())
                    state = t.tensor(state, dtype=t.float32).flatten()
                    total_reward += float(reward)

                    td3_train.store_transition(
                        {
                            "state": {"state": old_state.unsqueeze(0)},
                            "action": {"action": action},
                            "next_state": {"state": state.unsqueeze(0)},
                            "reward": float(reward),
                            "terminal": terminal or step == c.max_steps,
                        }
                    )
            # update
            if episode > 100:
                for i in range(step.get()):
                    td3_train.update()

            smoother.update(total_reward)
            step.reset()
            terminal = False

            if episode.get() % c.noise_interval != 0:
                # only log result without noise
                logger.info(f"Episode {episode} total reward={smoother.value:.2f}")

            if smoother.value > c.solved_reward:
                reward_fulfilled.count()
                if reward_fulfilled >= c.solved_repeat:
                    logger.info("Environment solved!")
                    return
            else:
                reward_fulfilled.reset()

        pytest.fail("TD3 Training failed.")
