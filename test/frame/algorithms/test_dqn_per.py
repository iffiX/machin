from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.dqn_per import DQNPer
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window

import pytest
import torch as t
import torch.nn as nn
import gym

from .utils import unwrap_time_limit, Smooth


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


class TestDQNPer(object):
    # configs and definitions
    @pytest.fixture(scope="class")
    def train_config(self, pytestconfig):
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
        c.solved_reward = 190
        c.solved_repeat = 5
        c.device = pytestconfig.get_option("gpu_device")
        return c

    @pytest.fixture(scope="function")
    def dqn_per(self, train_config):
        c = train_config
        q_net = smw(QNet(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        q_net_t = smw(QNet(c.observe_dim, c.action_num)
                      .to(c.device), c.device, c.device)
        dqn_per = DQNPer(q_net, q_net_t,
                         t.optim.Adam,
                         nn.MSELoss(reduction='sum'),
                         replay_device="cpu",
                         replay_size=c.replay_size)
        return dqn_per

    @pytest.fixture(scope="function")
    def dqn_per_vis(self, train_config, tmpdir):
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        q_net = smw(QNet(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        q_net_t = smw(QNet(c.observe_dim, c.action_num)
                      .to(c.device), c.device, c.device)
        dqn_per = DQNPer(q_net, q_net_t,
                         t.optim.Adam,
                         nn.MSELoss(reduction='sum'),
                         replay_device="cpu",
                         replay_size=c.replay_size,
                         visualize=True,
                         visualize_dir=str(tmp_dir))
        return dqn_per

    ########################################################################
    # Test for DQNPer criterion (mainly code coverage)
    ########################################################################
    def test_criterion(self, train_config):
        c = train_config
        q_net = smw(QNet(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        q_net_t = smw(QNet(c.observe_dim, c.action_num)
                      .to(c.device), c.device, c.device)
        with pytest.raises(RuntimeError, match="Criterion does not have the "
                                               "'reduction' property"):
            def criterion(a, b):
                return a - b
            _ = DQNPer(q_net, q_net_t,
                       t.optim.Adam,
                       criterion,
                       replay_device="cpu",
                       replay_size=c.replay_size,
                       mode="invalid_mode")

    ########################################################################
    # Test for DQNPer acting
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for DQNPer criticizing
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for DQNPer storage
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for DQNPer update
    ########################################################################
    def test_update(self, train_config, dqn_per_vis):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, 1], dtype=t.int)
        dqn_per_vis.store_episode([
            {"state": {"state": old_state},
             "action": {"action": action},
             "next_state": {"state": state},
             "reward": 0,
             "terminal": False}
            for _ in range(3)
        ])
        dqn_per_vis.update(update_value=True,
                           update_target=True,
                           concatenate_samples=True)
        dqn_per_vis.store_episode([
            {"state": {"state": old_state},
             "action": {"action": action},
             "next_state": {"state": state},
             "reward": 0,
             "terminal": False}
            for _ in range(3)
        ])
        dqn_per_vis.update(update_value=False,
                           update_target=False,
                           concatenate_samples=True)

    ########################################################################
    # Test for DQNPer save & load
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for DQNPer lr_scheduler
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for DQNPer full training.
    ########################################################################
    def test_full_train(self, train_config, dqn_per):
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
            state = t.tensor(env.reset(), dtype=t.float32, device=c.device)

            while not terminal and step <= c.max_steps:
                step.count()
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action = dqn_per.act_discrete_with_noise(
                        {"state": old_state.unsqueeze(0)}
                    )
                    state, reward, terminal, _ = env.step(action.item())
                    state = t.tensor(state, dtype=t.float32, device=c.device) \
                        .flatten()
                    total_reward += float(reward)

                    dqn_per.store_transition({
                        "state": {"state": old_state.unsqueeze(0)},
                        "action": {"action": action},
                        "next_state": {"state": state.unsqueeze(0)},
                        "reward": float(reward),
                        "terminal": terminal or step == c.max_steps
                    })

            # update
            if episode.get() > 100:
                for _ in range(step.get()):
                    dqn_per.update()

            smoother.update(total_reward)
            step.reset()
            terminal = False

            logger.info("Episode {} total reward={:.2f}"
                        .format(episode, smoother.value))

            if smoother.value > c.solved_reward:
                reward_fulfilled.count()
                if reward_fulfilled >= c.solved_repeat:
                    logger.info("Environment solved!")
                    return
            else:
                reward_fulfilled.reset()

        pytest.fail("DQNPer Training failed.")
