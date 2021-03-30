from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.dqn import DQN
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


class TestDQN(object):
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
        c.replay_size = 100000
        c.solved_reward = 150
        c.solved_repeat = 5
        return c

    @pytest.fixture(scope="function", params=["double"])
    def dqn(self, train_config, device, dtype, request):
        c = train_config
        q_net = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        q_net_t = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        dqn = DQN(
            q_net,
            q_net_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
            mode=request.param,
        )
        return dqn

    @pytest.fixture(scope="function", params=["double"])
    def dqn_vis(self, train_config, device, dtype, tmpdir, request):
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        q_net = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        q_net_t = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        dqn = DQN(
            q_net,
            q_net_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
            mode=request.param,
            visualize=True,
            visualize_dir=str(tmp_dir),
        )
        return dqn

    @pytest.fixture(scope="function")
    def dqn_lr(self, train_config, device, dtype):
        # not used for training, only used for testing apis
        c = train_config
        q_net = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        q_net_t = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)], logger=logger)
        with pytest.raises(TypeError, match="missing .+ positional argument"):
            _ = DQN(
                q_net,
                q_net_t,
                t.optim.Adam,
                nn.MSELoss(reduction="sum"),
                replay_device="cpu",
                replay_size=c.replay_size,
                lr_scheduler=LambdaLR,
            )
        dqn = DQN(
            q_net,
            q_net_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
            lr_scheduler=LambdaLR,
            lr_scheduler_args=((lr_func,),),
        )
        return dqn

    @pytest.fixture(scope="function", params=["double"])
    def dqn_train(self, train_config, request):
        c = train_config
        # cpu is faster for testing full training.
        q_net = smw(QNet(c.observe_dim, c.action_num), "cpu", "cpu")
        q_net_t = smw(QNet(c.observe_dim, c.action_num), "cpu", "cpu")
        dqn = DQN(
            q_net,
            q_net_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
            mode=request.param,
        )
        return dqn

    ########################################################################
    # Test for DQN modes (mainly code coverage)
    ########################################################################
    def test_mode(self, train_config, device, dtype):
        c = train_config
        q_net = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        q_net_t = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )

        with pytest.raises(ValueError, match="Unknown DQN mode"):
            _ = DQN(
                q_net,
                q_net_t,
                t.optim.Adam,
                nn.MSELoss(reduction="sum"),
                replay_device="cpu",
                replay_size=c.replay_size,
                mode="invalid_mode",
            )

        with pytest.raises(ValueError, match="Unknown DQN mode"):
            dqn = DQN(
                q_net,
                q_net_t,
                t.optim.Adam,
                nn.MSELoss(reduction="sum"),
                replay_device="cpu",
                replay_size=c.replay_size,
                mode="double",
            )

            old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
            action = t.zeros([1, 1], dtype=t.int)
            dqn.store_episode(
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
            dqn.mode = "invalid_mode"
            dqn.update(update_value=True, update_target=True, concatenate_samples=True)

    ########################################################################
    # Test for DQN acting
    ########################################################################
    def test_act(self, train_config, dqn, dtype):
        c = train_config
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        dqn.act_discrete({"state": state})
        dqn.act_discrete({"state": state}, True)
        dqn.act_discrete_with_noise({"state": state})
        dqn.act_discrete_with_noise({"state": state}, True)

    ########################################################################
    # Test for DQN criticizing
    ########################################################################
    def test__criticize(self, train_config, dqn, dtype):
        c = train_config
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        dqn._criticize({"state": state})
        dqn._criticize({"state": state}, True)

    ########################################################################
    # Test for DQN storage
    ########################################################################
    def test_store_step(self, train_config, dqn, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        dqn.store_transition(
            {
                "state": {"state": old_state},
                "action": {"action": action},
                "next_state": {"state": state},
                "reward": 0,
                "terminal": False,
            }
        )

    def test_store_episode(self, train_config, dqn, dtype):
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
        dqn.store_episode(episode)

    ########################################################################
    # Test for DQN update
    ########################################################################
    @pytest.mark.parametrize(
        "dqn_vis", ["vanilla", "fixed_target", "double"], indirect=True
    )
    def test_update(self, train_config, dqn_vis, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        dqn_vis.store_episode(
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
        dqn_vis.update(update_value=True, update_target=True, concatenate_samples=True)
        dqn_vis.store_episode(
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
        dqn_vis.update(
            update_value=False, update_target=False, concatenate_samples=True
        )

    ########################################################################
    # Test for DQN save & load
    ########################################################################
    def test_save_load(self, train_config, dqn, tmpdir):
        save_dir = tmpdir.make_numbered_dir()
        dqn.save(
            model_dir=str(save_dir), network_map={"qnet_target": "qnet_t"}, version=1000
        )
        dqn.load(
            model_dir=str(save_dir), network_map={"qnet_target": "qnet_t"}, version=1000
        )

    ########################################################################
    # Test for DQN lr_scheduler
    ########################################################################
    def test_lr_scheduler(self, train_config, dqn_lr):
        dqn_lr.update_lr_scheduler()

    ########################################################################
    # Test for DQN config & init
    ########################################################################
    def test_config_init(self, train_config):
        c = train_config
        config = DQN.generate_config({})
        config["frame_config"]["models"] = ["QNet", "QNet"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": c.observe_dim, "action_num": c.action_num}
        ] * 2
        dqn = DQN.init_from_config(config)

        old_state = state = t.zeros([1, c.observe_dim], dtype=t.float32)
        action = t.zeros([1, 1], dtype=t.int)
        dqn.store_episode(
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
        dqn.update()

    ########################################################################
    # Test for DQN full training.
    ########################################################################
    @pytest.mark.parametrize(
        "dqn_train", ["vanilla", "fixed_target", "double"], indirect=True
    )
    def test_full_train(self, train_config, dqn_train):
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
                    action = dqn_train.act_discrete_with_noise(
                        {"state": old_state.unsqueeze(0)}
                    )
                    state, reward, terminal, _ = env.step(action.item())
                    state = t.tensor(state, dtype=t.float32).flatten()
                    total_reward += float(reward)

                    dqn_train.store_transition(
                        {
                            "state": {"state": old_state.unsqueeze(0)},
                            "action": {"action": action},
                            "next_state": {"state": state.unsqueeze(0)},
                            "reward": float(reward),
                            "terminal": terminal or step == c.max_steps,
                        }
                    )

            # update
            if episode.get() > 100:
                for _ in range(step.get()):
                    dqn_train.update()

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

        pytest.fail("DQN Training failed.")
