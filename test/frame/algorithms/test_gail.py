from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.ppo import PPO
from machin.frame.algorithms.gail import GAIL
from machin.utils.learning_rate import gen_learning_rate_func
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical

import os
import pytest
import torch as t
import torch.nn as nn
import gym

from test.frame.algorithms.utils import unwrap_time_limit, Smooth
from test.util_fixtures import *
from test.util_platforms import linux_only


class Actor(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

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


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + 1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.action_num = action_num

    def forward(self, state, action: t.Tensor):
        d = t.relu(
            self.fc1(
                t.cat(
                    [state, action.type_as(state).view(-1, 1) / self.action_num], dim=1
                )
            )
        )
        d = t.relu(self.fc2(d))
        d = t.sigmoid(self.fc3(d))
        return d


class TestGAIL:
    # configs and definitions
    @pytest.fixture(scope="class")
    def train_config(self):
        disable_view_window()
        c = Config()
        # Note: online policy algorithms such as PPO and A2C does not
        # work well in Pendulum (reason unknown)
        # and MountainCarContinuous (sparse returns)
        c.env_name = "CartPole-v0"
        c.env = unwrap_time_limit(gym.make(c.env_name))
        c.observe_dim = 4
        c.action_num = 2
        c.max_episodes = 1000
        c.max_steps = 200
        c.replay_size = 10000
        c.solved_reward = 150
        c.solved_repeat = 5
        return c

    @pytest.fixture(scope="function")
    def gail(self, train_config, device, dtype):
        # not used for training, only used for testing apis
        c = train_config
        actor = smw(
            Actor(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        critic = smw(Critic(c.observe_dim).type(dtype).to(device), device, device)
        discriminator = smw(
            Discriminator(c.observe_dim, c.action_num).type(dtype).to(device),
            device,
            device,
        )
        ppo = PPO(
            actor,
            critic,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        gail = GAIL(
            discriminator,
            ppo,
            t.optim.Adam,
            expert_replay_device="cpu",
            expert_replay_size=c.replay_size,
        )
        return gail

    @pytest.fixture(scope="function")
    def gail_vis(self, train_config, device, dtype, tmpdir):
        # not used for training, only used for testing apis
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        actor = smw(
            Actor(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        critic = smw(Critic(c.observe_dim).type(dtype).to(device), device, device)
        discriminator = smw(
            Discriminator(c.observe_dim, c.action_num).type(dtype).to(device),
            device,
            device,
        )
        ppo = PPO(
            actor,
            critic,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
            visualize=True,
            visualize_dir=str(tmp_dir),
        )
        gail = GAIL(
            discriminator,
            ppo,
            t.optim.Adam,
            expert_replay_device="cpu",
            expert_replay_size=c.replay_size,
            visualize=True,
            visualize_dir=str(tmp_dir),
        )
        return gail

    @pytest.fixture(scope="function")
    def gail_lr(self, train_config, device, dtype):
        # not used for training, only used for testing apis
        c = train_config
        actor = smw(
            Actor(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        critic = smw(Critic(c.observe_dim).type(dtype).to(device), device, device)
        discriminator = smw(
            Discriminator(c.observe_dim, c.action_num).type(dtype).to(device),
            device,
            device,
        )
        ppo = PPO(
            actor,
            critic,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)], logger=logger)
        with pytest.raises(TypeError, match="missing .+ positional argument"):
            _ = GAIL(
                discriminator,
                ppo,
                t.optim.Adam,
                expert_replay_device="cpu",
                expert_replay_size=c.replay_size,
                lr_scheduler=LambdaLR,
            )

        gail = GAIL(
            discriminator,
            ppo,
            t.optim.Adam,
            expert_replay_device="cpu",
            expert_replay_size=c.replay_size,
            lr_scheduler=LambdaLR,
            lr_scheduler_args=((lr_func,),),
        )
        return gail

    @pytest.fixture(scope="function")
    def gail_train(self, train_config):
        c = train_config
        actor = smw(Actor(c.observe_dim, c.action_num), "cpu", "cpu")
        critic = smw(Critic(c.observe_dim), "cpu", "cpu")
        discriminator = smw(Discriminator(c.observe_dim, c.action_num), "cpu", "cpu")
        ppo = PPO(
            actor,
            critic,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        gail = GAIL(
            discriminator,
            ppo,
            t.optim.Adam,
            expert_replay_device="cpu",
            expert_replay_size=c.replay_size,
        )
        return gail

    ########################################################################
    # Test for GAIL acting
    ########################################################################
    def test_act(self, train_config, gail, dtype):
        c = train_config
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        gail.act({"state": state})

    ########################################################################
    # Test for GAIL discriminating
    ########################################################################
    def test__discriminate(self, train_config, gail, dtype):
        c = train_config
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        gail._discriminate({"state": state}, {"action": action})

    ########################################################################
    # Test for GAIL storage
    ########################################################################
    def test_store_episode(self, train_config, gail, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        episode = [
            {
                "state": {"state": old_state},
                "action": {"action": action},
                "next_state": {"state": state},
                "reward": 0,
                "terminal": False,
            }
            for _ in range(3)
        ]
        gail.store_episode(episode)

    def test_store_expert_episode(self, train_config, gail, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        episode = [
            {"state": {"state": old_state}, "action": {"action": action},}
            for _ in range(3)
        ]
        gail.store_expert_episode(episode)

    ########################################################################
    # Test for GAIL update
    ########################################################################
    def test_update(self, train_config, gail_vis, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        expert_episode = [
            {"state": {"state": old_state}, "action": {"action": action}}
            for _ in range(3)
        ]
        episode = [
            {
                "state": {"state": old_state},
                "action": {"action": action},
                "next_state": {"state": state},
                "reward": 0,
                "terminal": False,
            }
            for _ in range(3)
        ]
        gail_vis.store_episode(episode)
        gail_vis.store_expert_episode(expert_episode)
        gail_vis.update(
            update_value=True,
            update_policy=True,
            update_discriminator=True,
            concatenate_samples=True,
        )

        gail_vis.store_episode(episode)
        gail_vis.update(
            update_value=False,
            update_policy=False,
            update_discriminator=False,
            concatenate_samples=True,
        )

    ########################################################################
    # Test for GAIL save & load
    ########################################################################
    # Skipped, it is the same as base

    ########################################################################
    # Test for GAIL lr_scheduler
    ########################################################################
    def test_lr_scheduler(self, train_config, gail_lr, dtype):
        gail_lr.update_lr_scheduler()

    ########################################################################
    # Test for GAIL config & init
    ########################################################################
    def test_config_init(self, train_config, tmpdir, archives):
        dir = tmpdir.make_numbered_dir()
        t.save(
            archives["gail"].load().item("expert_trajectories"),
            os.path.join(dir, "trajectory.data"),
        )

        c = train_config
        config = GAIL.generate_config({})
        config["frame_config"]["PPO_config"]["frame_config"]["models"] = [
            "Actor",
            "Critic",
        ]
        config["frame_config"]["PPO_config"]["frame_config"]["model_kwargs"] = [
            {"state_dim": c.observe_dim, "action_num": c.action_num},
            {"state_dim": c.observe_dim},
        ]
        config["frame_config"]["models"] = ["Discriminator"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": c.observe_dim, "action_num": c.action_num},
        ]
        config["frame_config"]["expert_trajectory_path"] = os.path.join(
            dir, "trajectory.data"
        )
        gail = GAIL.init_from_config(config)

        old_state = state = t.zeros([1, c.observe_dim], dtype=t.float32)
        action = t.zeros([1, 1], dtype=t.float32)
        expert_episode = [
            {"state": {"state": old_state}, "action": {"action": action}}
            for _ in range(3)
        ]
        episode = [
            {
                "state": {"state": old_state},
                "action": {"action": action},
                "next_state": {"state": state},
                "reward": 0,
                "terminal": False,
            }
            for _ in range(3)
        ]
        gail.store_episode(episode)
        gail.store_expert_episode(expert_episode)
        gail.update()

    ########################################################################
    # Test for GAIL full training.
    ########################################################################
    @linux_only
    def test_full_train(self, train_config, gail_train, archives):
        c = train_config
        for expert_episode in archives["gail"].load().item("expert_trajectories"):
            gail_train.store_expert_episode(expert_episode)

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

            tmp_observations = []
            while not terminal and step <= c.max_steps:
                step.count()
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action = gail_train.act({"state": old_state.unsqueeze(0)})[0]
                    state, reward, terminal, _ = env.step(action.item())
                    state = t.tensor(state, dtype=t.float32).flatten()
                    total_reward += float(reward)

                    tmp_observations.append(
                        {
                            "state": {"state": old_state.unsqueeze(0)},
                            "action": {"action": action},
                            "next_state": {"state": state.unsqueeze(0)},
                            "reward": float(reward),
                            "terminal": terminal or step == c.max_steps,
                        }
                    )

            # update
            gail_train.store_episode(tmp_observations)
            gail_train.update()

            smoother.update(total_reward)
            step.reset()
            terminal = False

            logger.info(f"Episode {episode} total reward={smoother.value:.2f}")

            if smoother.value > c.solved_reward:
                reward_fulfilled.count()
                if reward_fulfilled >= c.solved_repeat:
                    logger.info("Environment solved!")
                    return
            else:
                reward_fulfilled.reset()

        pytest.fail("GAIL Training failed.")
