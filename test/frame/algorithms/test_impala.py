from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.impala import IMPALA
from machin.frame.helpers.servers import model_server_helper
from machin.utils.helper_classes import Counter
from machin.utils.learning_rate import gen_learning_rate_func
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical

import os
import torch as t
import torch.nn as nn
import gym

from .utils import unwrap_time_limit, Smooth
from test.util_run_multi import *
from test.util_fixtures import *


class Actor(nn.Module):
    def __init__(self, state_dim, action_num):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state, action=None):
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
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


class TestIMPALA(object):
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
    c.max_episodes = 2000
    c.max_steps = 200
    c.replay_size = 10000
    c.solved_reward = 150
    c.solved_repeat = 5

    @staticmethod
    def impala(device, dtype, use_lr_sch=False):
        c = TestIMPALA.c
        actor = smw(Actor(c.observe_dim, c.action_num)
                    .type(dtype).to(device), device, device)
        critic = smw(Critic(c.observe_dim)
                     .type(dtype).to(device), device, device)
        servers = model_server_helper(model_num=1)
        world = get_world()
        # process 0 and 1 will be workers, and 2 will be trainer
        impala_group = world.create_rpc_group("impala", ["0", "1", "2"])

        if use_lr_sch:
            lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)],
                                             logger=default_logger)
            impala = IMPALA(actor, critic,
                            t.optim.Adam,
                            nn.MSELoss(reduction='sum'),
                            impala_group,
                            servers,
                            lr_scheduler=LambdaLR,
                            lr_scheduler_args=((lr_func,), (lr_func,)))
        else:
            impala = IMPALA(actor, critic,
                            t.optim.Adam,
                            nn.MSELoss(reduction='sum'),
                            impala_group,
                            servers)
        return impala

    ########################################################################
    # Test for IMPALA acting
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["device", "dtype"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_act(_, device, dtype):
        c = TestIMPALA.c
        impala = TestIMPALA.impala(device, dtype)

        state = t.zeros([1, c.observe_dim], dtype=dtype)
        impala.act({"state": state})
        return True

    ########################################################################
    # Test for IMPALA action evaluation
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["device", "dtype"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_eval_action(_, device, dtype):
        c = TestIMPALA.c
        impala = TestIMPALA.impala(device, dtype)

        state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        impala._eval_act({"state": state}, {"action": action})
        return True

    ########################################################################
    # Test for IMPALA criticizing
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["device", "dtype"],
               timeout=180)
    @WorldTestBase.setup_world
    def test__criticize(_, device, dtype):
        c = TestIMPALA.c
        impala = TestIMPALA.impala(device, dtype)

        state = t.zeros([1, c.observe_dim], dtype=dtype)
        impala._criticize({"state": state})
        return True

    ########################################################################
    # Test for IMPALA storage
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["device", "dtype"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_store_step(_, device, dtype):
        c = TestIMPALA.c
        impala = TestIMPALA.impala(device, dtype)

        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)

        with pytest.raises(NotImplementedError):
            impala.store_transition({
                "state": {"state": old_state},
                "action": {"action": action},
                "next_state": {"state": state},
                "reward": 0,
                "action_log_prob": 0.1,
                "terminal": False
            })
        return True

    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["device", "dtype"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_store_episode(_, device, dtype):
        c = TestIMPALA.c
        impala = TestIMPALA.impala(device, dtype)

        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        episode = [
            {"state": {"state": old_state},
             "action": {"action": action},
             "next_state": {"state": state},
             "reward": 0,
             "action_log_prob": 0.1,
             "terminal": False}
            for _ in range(3)
        ]
        impala.store_episode(episode)
        return True

    ########################################################################
    # Test for IMPALA update
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["device", "dtype"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_update(rank, device, dtype):
        c = TestIMPALA.c
        impala = TestIMPALA.impala(device, dtype)

        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        if rank == 0:
            # episode length = 3
            impala.store_episode([
                {"state": {"state": old_state},
                 "action": {"action": action},
                 "next_state": {"state": state},
                 "reward": 0,
                 "action_log_prob": 0.1,
                 "terminal": False}
                for _ in range(3)
            ])
        elif rank == 1:
            # episode length = 2
            impala.store_episode([
                {"state": {"state": old_state},
                 "action": {"action": action},
                 "next_state": {"state": state},
                 "reward": 0,
                 "action_log_prob": 0.1,
                 "terminal": False}
                for _ in range(2)
            ])
        if rank == 2:
            sleep(2)
            impala.update(update_value=True,
                          update_target=True,
                          concatenate_samples=True)
        return True

    ########################################################################
    # Test for IMPALA save & load
    ########################################################################
    # Skipped, it is the same as base framework

    ########################################################################
    # Test for IMPALA lr_scheduler
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["device", "dtype"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_lr_scheduler(_, device, dtype):
        impala = TestIMPALA.impala(device, dtype)

        impala.update_lr_scheduler()
        return True

    ########################################################################
    # Test for IMPALA config & init
    ########################################################################
    @staticmethod
    @run_multi(timeout=180)
    @WorldTestBase.setup_world
    def test_config_init(_):
        config = IMPALA.generate_config({})
        config["frame_config"]["models"] = ["Actor", "Critic"]
        config["frame_config"]["model_kwargs"] = [{"state_dim": 4,
                                                   "action_num": 2},
                                                  {"state_dim": 4}]
        _impala = IMPALA.init_from_config(config)

    ########################################################################
    # Test for IMPALA full training.
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               timeout=1800)
    @WorldTestBase.setup_world
    def test_full_train(rank):
        c = TestIMPALA.c
        impala = TestIMPALA.impala("cpu", t.float32)

        # perform manual syncing to decrease the number of rpc calls
        impala.set_sync(False)

        # begin training
        episode, step = Counter(), Counter()
        reward_fulfilled = Counter()
        smoother = Smooth()
        terminal = False

        env = c.env
        world = get_world()
        all_group = world.create_rpc_group("all", ["0", "1", "2"])
        all_group.pair("{}_running".format(rank), True)
        default_logger.info("{}, pid {}".format(rank, os.getpid()))
        if rank == 0:
            all_group.pair("episode", episode)

        if rank in (0, 1):
            while episode < c.max_episodes:
                # wait for trainer to keep up
                sleep(0.2)
                episode.count()

                # batch size = 1
                total_reward = 0
                state = t.tensor(env.reset(), dtype=t.float32)

                impala.manual_sync()
                tmp_observations = []
                while not terminal and step <= c.max_steps:
                    step.count()
                    with t.no_grad():
                        old_state = state
                        action, action_log_prob, *_ = impala.act(
                            {"state": old_state.unsqueeze(0)})
                        state, reward, terminal, _ = env.step(action.item())
                        state = t.tensor(state, dtype=t.float32).flatten()
                        total_reward += float(reward)

                        tmp_observations.append({
                            "state": {"state": old_state.unsqueeze(0)},
                            "action": {"action": action},
                            "next_state": {"state": state.unsqueeze(0)},
                            "reward": float(reward),
                            "action_log_prob": action_log_prob.item(),
                            "terminal": terminal or step == c.max_steps
                        })
                impala.store_episode(tmp_observations)

                smoother.update(total_reward)
                step.reset()
                terminal = False

                default_logger.info("Process {} Episode {} "
                                    "total reward={:.2f}"
                                    .format(rank, episode, smoother.value))

                if smoother.value > c.solved_reward:
                    reward_fulfilled.count()
                    if reward_fulfilled >= c.solved_repeat:
                        default_logger.info("Environment solved!")

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
            # wait for some samples
            # Note: the number of entries in buffer means "episodes"
            # rather than steps here!
            while impala.replay_buffer.all_size() < 5:
                sleep(0.1)
            while (all_group.is_paired("0_running") or
                   all_group.is_paired("1_running")):
                impala.update()
            return True

        raise RuntimeError("IMPALA Training failed.")
