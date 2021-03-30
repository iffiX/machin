from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.rainbow import RAINBOW
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window

import pytest
import torch as t
import torch.nn as nn
import gym

from .utils import unwrap_time_limit, Smooth
from test.util_fixtures import *


class QNet(nn.Module):
    # this test setup lacks the noisy linear layer and dueling structure.
    def __init__(self, state_dim, action_num, atom_num=10):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num * atom_num)
        self.action_num = action_num
        self.atom_num = atom_num

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return t.softmax(self.fc3(a).view(-1, self.action_num, self.atom_num), dim=-1)


class TestRAINBOW(object):
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
        # maximum and minimum of reward value
        # since reward is 1 for every step, maximum q value should be
        # below 20(reward_future_steps) * (1 + discount ** n_steps) < 40
        c.value_max = 40
        c.value_min = 0
        c.reward_future_steps = 20
        c.max_episodes = 1000
        c.max_steps = 200
        c.replay_size = 100000

        # RAINBOW is not very stable (without dueling and noisy linear)
        # compared to other DQNs
        c.solved_reward = 150
        c.solved_repeat = 5
        return c

    @pytest.fixture(scope="function")
    def rainbow(self, train_config, device, dtype):
        c = train_config
        q_net = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        q_net_t = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        rainbow = RAINBOW(
            q_net,
            q_net_t,
            t.optim.Adam,
            c.value_min,
            c.value_max,
            reward_future_steps=c.reward_future_steps,
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        return rainbow

    @pytest.fixture(scope="function")
    def rainbow_vis(self, train_config, device, dtype, tmpdir):
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        q_net = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        q_net_t = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        rainbow = RAINBOW(
            q_net,
            q_net_t,
            t.optim.Adam,
            c.value_min,
            c.value_max,
            reward_future_steps=c.reward_future_steps,
            replay_device="cpu",
            replay_size=c.replay_size,
            visualize=True,
            visualize_dir=str(tmp_dir),
        )
        return rainbow

    @pytest.fixture(scope="function")
    def rainbow_train(self, train_config):
        c = train_config
        # cpu is faster for testing full training.
        q_net = smw(QNet(c.observe_dim, c.action_num), "cpu", "cpu")
        q_net_t = smw(QNet(c.observe_dim, c.action_num), "cpu", "cpu")
        rainbow = RAINBOW(
            q_net,
            q_net_t,
            t.optim.Adam,
            c.value_min,
            c.value_max,
            reward_future_steps=c.reward_future_steps,
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        return rainbow

    ########################################################################
    # Test for RAINBOW acting
    ########################################################################
    def test_act(self, train_config, rainbow, dtype):
        c = train_config
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        rainbow.act_discrete({"state": state})
        rainbow.act_discrete({"state": state}, True)
        rainbow.act_discrete_with_noise({"state": state})
        rainbow.act_discrete_with_noise({"state": state}, True)

    ########################################################################
    # Test for RAINBOW criticizing
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for RAINBOW storage
    ########################################################################
    def test_store_step(self, train_config, rainbow, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        rainbow.store_transition(
            {
                "state": {"state": old_state},
                "action": {"action": action},
                "next_state": {"state": state},
                "reward": 0,
                "value": 0,
                "terminal": False,
            }
        )

    def test_store_episode(self, train_config, rainbow, dtype):
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
        rainbow.store_episode(episode)

    ########################################################################
    # Test for RAINBOW update
    ########################################################################
    def test_update(self, train_config, rainbow_vis, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        rainbow_vis.store_episode(
            [
                {
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": 0,
                    "terminal": False,
                }
                for _ in range(3)
            ]
        )
        rainbow_vis.update(
            update_value=True, update_target=True, concatenate_samples=True
        )
        rainbow_vis.store_episode(
            [
                {
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": 0,
                    "terminal": False,
                }
                for _ in range(3)
            ]
        )
        rainbow_vis.update(
            update_value=False, update_target=False, concatenate_samples=True
        )

    ########################################################################
    # Test for RAINBOW save & load
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for RAINBOW lr_scheduler
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for RAINBOW config & init
    ########################################################################
    def test_config_init(self, train_config):
        c = train_config
        config = RAINBOW.generate_config({})
        config["frame_config"]["models"] = ["QNet", "QNet"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": c.observe_dim, "action_num": c.action_num}
        ] * 2
        rainbow = RAINBOW.init_from_config(config)

        old_state = state = t.zeros([1, c.observe_dim], dtype=t.float32)
        action = t.zeros([1, 1], dtype=t.int)
        rainbow.store_episode(
            [
                {
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": 0,
                    "terminal": False,
                }
                for _ in range(3)
            ]
        )
        rainbow.update()

    ########################################################################
    # Test for RAINBOW full training.
    ########################################################################
    def test_full_train(self, train_config, rainbow_train):
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

            tmp_observations = []
            while not terminal and step <= c.max_steps:
                step.count()
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action = rainbow_train.act_discrete_with_noise(
                        {"state": old_state.unsqueeze(0)}
                    )
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

            rainbow_train.store_episode(tmp_observations)
            # update
            if episode.get() > 100:
                for _ in range(step.get()):
                    rainbow_train.update()

            smoother.update(total_reward)
            step.reset()
            terminal = False

            logger.info(
                "Episode {} total reward={:.2f}".format(episode, smoother.value)
            )

            if smoother.value > c.solved_reward:
                reward_fulfilled.count()
                if reward_fulfilled >= c.solved_repeat:
                    logger.info("Environment solved!")
                    return
            else:
                reward_fulfilled.reset()

        pytest.fail("RAINBOW Training failed.")
