from time import sleep
from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.apex import DQNApex, DDPGApex
from machin.frame.helpers.servers import model_server_helper
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
from machin.parallel.distributed import get_world

import pytest
import torch as t
import torch.nn as nn
import gym

from .utils import unwrap_time_limit, Smooth
from test.util_run_multi import *


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


class TestDQNApex(object):
    # configs and definitions
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
    c.device = "cpu"

    @staticmethod
    def dqn_apex(rank):
        c = TestDQNApex.c
        q_net = smw(QNet(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        q_net_t = smw(QNet(c.observe_dim, c.action_num)
                      .to(c.device), c.device, c.device)
        servers = model_server_helper()
        world = get_world()
        # process 0 and 1 will be workers, and 2 will be trainer
        if rank in (0, 1):
            worker_group = world.create_rpc_group("worker", ["0", "1"])
        else:
            sleep(2)
            worker_group = world.get_rpc_group("worker", "0")
        dqn_per = DQNApex(q_net, q_net_t,
                          t.optim.Adam,
                          nn.MSELoss(reduction='sum'),
                          worker_group,
                          servers,
                          replay_device=c.device,
                          replay_size=c.replay_size)
        return dqn_per

    ########################################################################
    # Test for DQNApex acting
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_act(rank):
        c = TestDQNApex.c
        dqn_apex = TestDQNApex.dqn_apex(rank)
        state = t.zeros([1, c.observe_dim])
        dqn_apex.act_discreet({"state": state})
        dqn_apex.act_discreet({"state": state}, True)
        dqn_apex.act_discreet_with_noise({"state": state})
        dqn_apex.act_discreet_with_noise({"state": state}, True)
        return True

    ########################################################################
    # Test for DQNApex criticizing
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_criticize(rank):
        c = TestDQNApex.c
        dqn_apex = TestDQNApex.dqn_apex(rank)
        state = t.zeros([1, c.observe_dim])
        dqn_apex.criticize({"state": state})
        dqn_apex.criticize({"state": state}, True)
        return True

    ########################################################################
    # Test for DQNApex storage
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for DQNApex update
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_update(rank):
        c = TestDQNApex.c
        dqn_apex = TestDQNApex.dqn_apex(rank)
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, 1], dtype=t.int)
        if rank in (0, 1):
            dqn_apex.store_episode([
                {"state": {"state": old_state.clone()},
                 "action": {"action": action.clone()},
                 "next_state": {"state": state.clone()},
                 "reward": 0,
                 "terminal": False}
                for _ in range(3)
            ])
        if rank == 2:
            sleep(2)
            dqn_apex.update(update_value=True,
                            update_target=True,
                            concatenate_samples=True)
        return True

    ########################################################################
    # Test for DQNApex save & load
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for DQNApex lr_scheduler
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for DQNApex full training.
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_full_train(rank):
        c = TestDQNApex.c
        dqn_apex = TestDQNApex.dqn_apex(rank)

        # begin training
        episode, step = Counter(), Counter()
        reward_fulfilled = Counter()
        smoother = Smooth()
        terminal = False

        env = c.env
        world = get_world()
        all_group = world.create_rpc_group("all", ["0", "1", "2"])
        all_group.pair("0_running", True)
        all_group.pair("1_running", True)
        if rank in (0, 1):
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
                        action = dqn_apex.act_discreet_with_noise(
                            {"state": old_state.unsqueeze(0)}
                        )
                        state, reward, terminal, _ = env.step(action.item())
                        state = t.tensor(state, dtype=t.float32,
                                         device=c.device).flatten()
                        total_reward += float(reward)

                        dqn_apex.store_transition({
                            "state": {"state": old_state.unsqueeze(0).clone()},
                            "action": {"action": action.clone()},
                            "next_state": {"state": state.unsqueeze(0).clone()},
                            "reward": float(reward),
                            "terminal": terminal or step == c.max_steps
                        })

                smoother.update(total_reward)
                step.reset()
                terminal = False

                logger.info("Episode {} total reward={:.2f}"
                            .format(episode, smoother.value))

                if smoother.value > c.solved_reward:
                    reward_fulfilled.count()
                    if reward_fulfilled >= c.solved_repeat:
                        logger.info("Environment solved!")

                        all_group.unpair("{}_running".format(rank))
                        while (all_group.is_paired("0_running") or
                               all_group.is_paired("1_running")):
                            # wait for all workers to join
                            sleep(1)
                        # wait for trainer
                        sleep(5)
                        return True
                else:
                    reward_fulfilled.reset()
        else:
            while (all_group.is_paired("0_running") or
                    all_group.is_paired("1_running")):
                dqn_apex.update()
            return True

        raise RuntimeError("A3C Training failed.")
