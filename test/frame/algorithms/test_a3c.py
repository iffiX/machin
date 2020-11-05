from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.a3c import A3C
from machin.frame.helpers.servers import grad_server_helper
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
from torch.distributions import Categorical

import os
import torch as t
import torch.nn as nn
import gym

from .utils import unwrap_time_limit, Smooth
from test.util_run_multi import *


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


class TestA3C(object):
    # configs and definitions
    disable_view_window()
    c = Config()
    # Note: online policy algorithms such as PPO and A3C does not
    # work well in Pendulum (reason unknown)
    # and MountainCarContinuous (sparse returns)
    c.env_name = "CartPole-v0"
    c.env = unwrap_time_limit(gym.make(c.env_name))
    c.observe_dim = 4
    c.action_num = 2
    c.max_episodes = 3000
    c.max_steps = 200
    c.replay_size = 10000
    c.solved_reward = 190
    c.solved_repeat = 5

    @staticmethod
    def a3c():
        """
        Perform a 3d action.

        Args:
        """
        c = TestA3C.c
        actor = smw(Actor(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim)
                     .to(c.device), c.device, c.device)
        # in all test scenarios, all processes will be used as reducers
        servers = grad_server_helper(
            [lambda: Actor(c.observe_dim, c.action_num),
             lambda: Critic(c.observe_dim)],
            learning_rate=5e-3
        )
        a3c = A3C(actor, critic,
                  nn.MSELoss(reduction='sum'),
                  servers,
                  replay_device="cpu",
                  replay_size=c.replay_size)
        return a3c

    ########################################################################
    # Test for A3C acting
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["gpu"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_act(_, gpu):
        """
        Test if the gpu of the expected state.

        Args:
            _: (array): write your description
            gpu: (array): write your description
        """
        c = TestA3C.c
        c.device = gpu
        a3c = TestA3C.a3c()
        state = t.zeros([1, c.observe_dim])
        a3c.act({"state": state})
        return True

    ########################################################################
    # Test for A3C action evaluation
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["gpu"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_eval_action(_, gpu):
        """
        Evaluates the device.

        Args:
            _: (todo): write your description
            gpu: (todo): write your description
        """
        c = TestA3C.c
        c.device = gpu
        a3c = TestA3C.a3c()
        state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, 1], dtype=t.int)
        a3c._eval_act({"state": state}, {"action": action})
        return True

    ########################################################################
    # Test for A3C criticizing
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["gpu"],
               timeout=180)
    @WorldTestBase.setup_world
    def test__criticize(_, gpu):
        """
        Determine if the search criteria match.

        Args:
            _: (todo): write your description
            gpu: (todo): write your description
        """
        c = TestA3C.c
        c.device = gpu
        a3c = TestA3C.a3c()
        state = t.zeros([1, c.observe_dim])
        a3c._criticize({"state": state})
        return True

    ########################################################################
    # Test for A3C storage
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for A3C update
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["gpu"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_update(rank, gpu):
        """
        Perform the device.

        Args:
            rank: (int): write your description
            gpu: (float): write your description
        """
        c = TestA3C.c
        c.device = gpu
        a3c = TestA3C.a3c()
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, 1], dtype=t.int)

        begin = time()
        while time() - begin < 5:
            a3c.store_episode([
                {"state": {"state": old_state},
                 "action": {"action": action},
                 "next_state": {"state": state},
                 "reward": 0,
                 "terminal": False}
                for _ in range(3)
            ])
            a3c.update(update_value=True, update_policy=True,
                       update_target=True, concatenate_samples=True)
            sleep(0.01)

        if rank == 1:
            # pull the newest model
            a3c.manual_sync()
        return True

    ########################################################################
    # Test for A3C save & load
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for A3C lr_scheduler
    ########################################################################
    # Skipped, it is the same as A2C

    ########################################################################
    # Test for A3C full training.
    ########################################################################
    @staticmethod
    @pytest.mark.parametrize("gae_lambda", [0.0, 0.5, 1.0])
    @run_multi(expected_results=[True, True, True],
               pass_through=["gae_lambda", "gpu"],
               timeout=1800)
    @WorldTestBase.setup_world
    def test_full_train(rank, gae_lambda, gpu):
        """
        Perform a full training.

        Args:
            rank: (int): write your description
            gae_lambda: (todo): write your description
            gpu: (todo): write your description
        """
        c = TestA3C.c
        c.device = gpu
        a3c = TestA3C.a3c()
        a3c.set_sync(False)

        # begin training
        episode, step = Counter(), Counter()
        reward_fulfilled = Counter()
        smoother = Smooth()
        terminal = False

        env = c.env
        # for cpu usage viewing
        default_logger.info("{}, pid {}".format(rank, os.getpid()))
        while episode < c.max_episodes:
            episode.count()

            # batch size = 1
            total_reward = 0
            state = t.tensor(env.reset(), dtype=t.float32, device=c.device)

            a3c.manual_sync()
            tmp_observations = []
            while not terminal and step <= c.max_steps:
                step.count()
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action = a3c.act({"state": old_state.unsqueeze(0)})[0]
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
            a3c.store_episode(tmp_observations)
            a3c.update()

            smoother.update(total_reward)
            step.reset()
            terminal = False

            default_logger.info("Process {} Episode {} total reward={:.2f}"
                                .format(rank, episode, smoother.value))

            if smoother.value > c.solved_reward:
                reward_fulfilled.count()
                if reward_fulfilled >= c.solved_repeat:
                    default_logger.info("Environment solved!")
                    return True
            else:
                reward_fulfilled.reset()

        raise RuntimeError("A3C Training failed.")
