from machin.model.nets.base import static_module_wrapper as smw
from machin.model.algorithms.trpo import ActorDiscrete, ActorContinuous
from machin.frame.algorithms.trpo import TRPO
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window

import pytest
import torch as t
import torch.nn as nn
import gym

from test.frame.algorithms.utils import unwrap_time_limit, Smooth
from test.util_fixtures import *
from test.util_platforms import linux_only


class Actor(ActorDiscrete):
    def __init__(self, state_dim, action_num):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)
        self.input_module = self.fc1
        self.output_module = self.fc3

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
        return self.sample(probs, action)


class ActorC(ActorContinuous):
    def __init__(self, state_dim, action_dim):
        super().__init__(action_dim)
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.input_module = self.fc1
        self.output_module = self.fc3

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        mean = t.softmax(self.fc3(a), dim=1)
        return self.sample(mean, action)


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


class TestTRPO:
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
        c.max_episodes = 2000  # the actor learns a little bit slower
        c.max_steps = 200
        c.replay_size = 10000
        c.solved_reward = 150
        c.solved_repeat = 5
        return c

    @pytest.fixture(scope="function", params=["fim", "direct"])
    def trpo_vis_disc(self, train_config, device, dtype, tmpdir, request):
        # not used for training, only used for testing apis
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        actor = Actor(c.observe_dim, c.action_num).type(dtype).to(device)
        critic = smw(Critic(c.observe_dim).type(dtype).to(device), device, device)
        trpo = TRPO(
            actor,
            critic,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            hv_mode=request.param,
            replay_device="cpu",
            replay_size=c.replay_size,
            visualize=True,
            visualize_dir=str(tmp_dir),
        )
        return trpo

    @pytest.fixture(scope="function", params=["fim", "direct"])
    def trpo_vis_cont(self, train_config, device, dtype, tmpdir, request):
        # not used for training, only used for testing apis
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        actor = ActorC(c.observe_dim, c.action_num).type(dtype).to(device)
        critic = smw(Critic(c.observe_dim).type(dtype).to(device), device, device)
        trpo = TRPO(
            actor,
            critic,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            hv_mode=request.param,
            replay_device="cpu",
            replay_size=c.replay_size,
            visualize=True,
            visualize_dir=str(tmp_dir),
        )
        return trpo

    @pytest.fixture(scope="function", params=["fim", "direct"])
    def trpo_train(self, train_config, request):
        c = train_config
        actor = Actor(c.observe_dim, c.action_num)
        critic = smw(Critic(c.observe_dim), "cpu", "cpu")
        trpo = TRPO(
            actor,
            critic,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            hv_mode=request.param,
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        return trpo

    ########################################################################
    # Test for TRPO acting
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for TRPO action evaluation
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for TRPO criticizing
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for TRPO storage
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for TRPO update
    ########################################################################
    def test_update_discrete(self, train_config, trpo_vis_disc, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        trpo_vis_disc.store_episode(
            [
                {
                    "state": {"state": old_state},
                    "action": {"action": t.ones([1, 1], dtype=dtype) * idx % 2},
                    "next_state": {"state": state},
                    "reward": idx * 0.1,
                    "terminal": False,
                }
                for idx in range(3)
            ]
        )
        trpo_vis_disc.update(
            update_value=True, update_policy=True, concatenate_samples=True,
        )
        trpo_vis_disc.store_episode(
            [
                {
                    "state": {"state": old_state},
                    "action": {"action": t.ones([1, 1], dtype=dtype) * idx % 2},
                    "next_state": {"state": state},
                    "reward": idx * 0.1,
                    "terminal": False,
                }
                for idx in range(3)
            ]
        )
        trpo_vis_disc.update(
            update_value=False, update_policy=False, concatenate_samples=True,
        )

    def test_update_continuous(self, train_config, trpo_vis_cont, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        trpo_vis_cont.store_episode(
            [
                {
                    "state": {"state": old_state},
                    "action": {"action": t.rand([1, c.action_num], dtype=dtype)},
                    "next_state": {"state": state},
                    "reward": idx * 0.1,
                    "terminal": False,
                }
                for idx in range(3)
            ]
        )
        trpo_vis_cont.update(
            update_value=True, update_policy=True, concatenate_samples=True,
        )
        trpo_vis_cont.store_episode(
            [
                {
                    "state": {"state": old_state},
                    "action": {"action": t.ones([1, c.action_num], dtype=dtype)},
                    "next_state": {"state": state},
                    "reward": idx * 0.1,
                    "terminal": False,
                }
                for idx in range(3)
            ]
        )
        trpo_vis_cont.update(
            update_value=False, update_policy=False, concatenate_samples=True,
        )

    ########################################################################
    # Test for TRPO save & load
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for TRPO lr_scheduler
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for TRPO config & init
    ########################################################################
    def test_config_init(self, train_config):
        c = train_config
        config = TRPO.generate_config({})
        config["frame_config"]["models"] = ["Actor", "Critic"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": c.observe_dim, "action_num": c.action_num},
            {"state_dim": c.observe_dim},
        ]
        trpo = TRPO.init_from_config(config)

        old_state = state = t.zeros([1, c.observe_dim], dtype=t.float32)
        trpo.store_episode(
            [
                {
                    "state": {"state": old_state},
                    "action": {"action": t.ones([1, 1], dtype=t.float32) * idx % 2},
                    "next_state": {"state": state},
                    "reward": idx * 0.1,
                    "terminal": False,
                }
                for idx in range(3)
            ]
        )
        trpo.update()

    ########################################################################
    # Test for TRPO full training.
    ########################################################################
    @linux_only
    @pytest.mark.parametrize("gae_lambda", [0.0, 0.5, 1.0])
    def test_full_train(self, train_config, trpo_train, gae_lambda):
        c = train_config
        trpo_train.gae_lambda = gae_lambda

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
                    action = trpo_train.act({"state": old_state.unsqueeze(0)})[0]
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
            trpo_train.store_episode(tmp_observations)
            trpo_train.update()

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

        pytest.fail("TRPO Training failed.")
