from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.a2c import A2C
from machin.utils.learning_rate import gen_learning_rate_func
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical

import pytest
import torch as t
import torch.nn as nn
import gym

from .utils import unwrap_time_limit, Smooth
from test.util_run_multi import gpu


class Actor(nn.Module):
    def __init__(self, state_dim, action_num):
        """
        Initialize the gradient.

        Args:
            self: (todo): write your description
            state_dim: (int): write your description
            action_num: (int): write your description
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state, action=None):
        """
        Perform forward forward computation

        Args:
            self: (todo): write your description
            state: (todo): write your description
            action: (str): write your description
        """
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = (action
               if action is not None
               else dist.sample())
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            state_dim: (int): write your description
        """
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        """
        Calculate forward.

        Args:
            self: (todo): write your description
            state: (todo): write your description
        """
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


class TestA2C(object):
    # configs and definitions
    @pytest.fixture(scope="class")
    def train_config(self, gpu):
        """
        Train a device configuration

        Args:
            self: (todo): write your description
            gpu: (todo): write your description
        """
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
        c.solved_reward = 190
        c.solved_repeat = 5
        c.device = gpu
        return c

    @pytest.fixture(scope="function")
    def a2c(self, train_config):
        """
        Create a device with the device.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
        """
        c = train_config
        actor = smw(Actor(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim)
                     .to(c.device), c.device, c.device)
        a2c = A2C(actor, critic,
                  t.optim.Adam,
                  nn.MSELoss(reduction='sum'),
                  replay_device=c.device,
                  replay_size=c.replay_size)
        return a2c

    @pytest.fixture(scope="function")
    def a2c_vis(self, train_config, tmpdir):
        """
        Make a device.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            tmpdir: (todo): write your description
        """
        # not used for training, only used for testing apis
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        actor = smw(Actor(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim)
                     .to(c.device), c.device, c.device)
        a2c = A2C(actor, critic,
                  t.optim.Adam,
                  nn.MSELoss(reduction='sum'),
                  replay_device=c.device,
                  replay_size=c.replay_size,
                  visualize=True,
                  visualize_dir=str(tmp_dir))
        return a2c

    @pytest.fixture(scope="function")
    def a2c_lr(self, train_config):
        """
        Perform a learning rate.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
        """
        # not used for training, only used for testing apis
        c = train_config
        actor = smw(Actor(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim)
                     .to(c.device), c.device, c.device)
        lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)],
                                         logger=logger)
        with pytest.raises(TypeError, match="missing .+ positional argument"):
            _ = A2C(actor, critic,
                    t.optim.Adam,
                    nn.MSELoss(reduction='sum'),
                    replay_device=c.device,
                    replay_size=c.replay_size,
                    lr_scheduler=LambdaLR)
        a2c = A2C(actor, critic,
                  t.optim.Adam,
                  nn.MSELoss(reduction='sum'),
                  replay_device=c.device,
                  replay_size=c.replay_size,
                  lr_scheduler=LambdaLR,
                  lr_scheduler_args=((lr_func,), (lr_func,)))
        return a2c

    ########################################################################
    # Test for A2C acting
    ########################################################################
    def test_act(self, train_config, a2c):
        """
        Test if train of the test_config.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            a2c: (todo): write your description
        """
        c = train_config
        state = t.zeros([1, c.observe_dim])
        a2c.act({"state": state})

    ########################################################################
    # Test for A2C action evaluation
    ########################################################################
    def test_eval_action(self, train_config, a2c):
        """
        Evaluates the environment.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            a2c: (todo): write your description
        """
        c = train_config
        state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, 1], dtype=t.int)
        a2c._eval_act({"state": state}, {"action": action})

    ########################################################################
    # Test for A2C criticizing
    ########################################################################
    def test__criticize(self, train_config, a2c):
        """
        Test for train_config.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            a2c: (todo): write your description
        """
        c = train_config
        state = t.zeros([1, c.observe_dim])
        a2c._criticize({"state": state})

    ########################################################################
    # Test for A2C storage
    ########################################################################
    def test_store_step(self, train_config, a2c):
        """
        Test for a single step.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            a2c: (todo): write your description
        """
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, 1], dtype=t.int)
        a2c.store_transition({
            "state": {"state": old_state},
            "action": {"action": action},
            "next_state": {"state": state},
            "reward": 0,
            "value": 0,
            "gae": 0,
            "terminal": False
        })

    @pytest.mark.parametrize("gae_lambda", [0.0, 0.5, 1.0])
    def test_store_episode(self, train_config, a2c, gae_lambda):
        """
        Store the episode.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            a2c: (todo): write your description
            gae_lambda: (todo): write your description
        """
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, 1], dtype=t.int)
        episode = [
            {"state": {"state": old_state},
             "action": {"action": action},
             "next_state": {"state": state},
             "reward": 0,
             "terminal": False}
            for _ in range(3)
        ]
        a2c.gae_lambda = gae_lambda
        a2c.store_episode(episode)

    ########################################################################
    # Test for A2C update
    ########################################################################
    def test_update(self, train_config, a2c_vis):
        """
        Perform the test test.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            a2c_vis: (todo): write your description
        """
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, 1], dtype=t.int)
        a2c_vis.store_episode([
            {"state": {"state": old_state},
             "action": {"action": action},
             "next_state": {"state": state},
             "reward": 0,
             "terminal": False}
            for _ in range(3)
        ])
        a2c_vis.update(update_value=True, update_policy=True,
                       update_target=True, concatenate_samples=True)
        a2c_vis.entropy_weight = 1e-3
        a2c_vis.store_episode([
            {"state": {"state": old_state},
             "action": {"action": action},
             "next_state": {"state": state},
             "reward": 0,
             "terminal": False}
            for _ in range(3)
        ])
        a2c_vis.update(update_value=False, update_policy=False,
                       update_target=False, concatenate_samples=True)

    ########################################################################
    # Test for A2C save & load
    ########################################################################
    # Skipped, it is the same as base

    ########################################################################
    # Test for A2C lr_scheduler
    ########################################################################
    def test_lr_scheduler(self, train_config, a2c_lr):
        """
        Sets the learning rate.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            a2c_lr: (todo): write your description
        """
        a2c_lr.update_lr_scheduler()

    ########################################################################
    # Test for A2C full training.
    ########################################################################
    @pytest.mark.parametrize("gae_lambda", [0.0, 0.5, 1.0])
    def test_full_train(self, train_config, a2c, gae_lambda):
        """
        Perform the environment.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            a2c: (todo): write your description
            gae_lambda: (todo): write your description
        """
        c = train_config
        a2c.gae_lambda = gae_lambda

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

            tmp_observations = []
            while not terminal and step <= c.max_steps:
                step.count()
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action = a2c.act({"state": old_state.unsqueeze(0)})[0]
                    state, reward, terminal, _ = env.step(action.item())
                    state = t.tensor(state, dtype=t.float32, device=c.device) \
                        .flatten()
                    total_reward += float(reward)

                    tmp_observations.append({
                        "state": {"state": old_state.unsqueeze(0)},
                        "action": {"action": action},
                        "next_state": {"state": state.unsqueeze(0)},
                        "reward": float(reward),
                        "terminal": terminal or step == c.max_steps
                    })

            # update
            a2c.store_episode(tmp_observations)
            a2c.update()

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

        pytest.fail("A2C Training failed.")
