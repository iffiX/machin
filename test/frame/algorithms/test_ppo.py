from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.ppo import PPO
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
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
        Initialize the graph.

        Args:
            self: (todo): write your description
            state_dim: (int): write your description
        """
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

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


class TestPPO(object):
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
    def ppo(self, train_config):
        """
        Perform a device.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
        """
        c = train_config
        actor = smw(Actor(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim)
                     .to(c.device), c.device, c.device)
        ppo = PPO(actor, critic,
                  t.optim.Adam,
                  nn.MSELoss(reduction='sum'),
                  replay_device="cpu",
                  replay_size=c.replay_size)
        return ppo

    @pytest.fixture(scope="function")
    def ppo_vis(self, train_config, tmpdir):
        """
        Initialize the device.

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
        ppo = PPO(actor, critic,
                  t.optim.Adam,
                  nn.MSELoss(reduction='sum'),
                  replay_device="cpu",
                  replay_size=c.replay_size,
                  visualize=True,
                  visualize_dir=str(tmp_dir))
        return ppo

    ########################################################################
    # Test for PPO acting
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for PPO action evaluation
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for PPO criticizing
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for PPO storage
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for PPO update
    ########################################################################
    def test_update(self, train_config, ppo_vis):
        """
        Updates the test update.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            ppo_vis: (todo): write your description
        """
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, 1])
        ppo_vis.store_episode([
            {"state": {"state": old_state},
             "action": {"action": action},
             "next_state": {"state": state},
             "reward": 0,
             "terminal": False}
            for _ in range(3)
        ])
        ppo_vis.update(update_value=True, update_policy=True,
                       update_target=True, concatenate_samples=True)
        ppo_vis.entropy_weight = 1e-3
        ppo_vis.store_episode([
            {"state": {"state": old_state},
             "action": {"action": action},
             "next_state": {"state": state},
             "reward": 0,
             "terminal": False}
            for _ in range(3)
        ])
        ppo_vis.update(update_value=False, update_policy=False,
                       update_target=False, concatenate_samples=True)

    ########################################################################
    # Test for PPO save & load
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for PPO lr_scheduler
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for PPO full training.
    ########################################################################
    @pytest.mark.parametrize("gae_lambda", [0.0, 0.5, 1.0])
    def test_full_train(self, train_config, ppo, gae_lambda):
        """
        Perform a full training.

        Args:
            self: (todo): write your description
            train_config: (todo): write your description
            ppo: (todo): write your description
            gae_lambda: (todo): write your description
        """
        c = train_config
        ppo.gae_lambda = gae_lambda

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
                    action = ppo.act({"state": old_state.unsqueeze(0)})[0]
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
            ppo.store_episode(tmp_observations)
            ppo.update()

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

        pytest.fail("PPO Training failed.")
