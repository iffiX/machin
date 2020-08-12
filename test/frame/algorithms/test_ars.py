from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.ars import ARS, RunningStat
from machin.frame.helpers.servers import model_server_helper
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.utils.learning_rate import gen_learning_rate_func
from machin.env.utils.openai_gym import disable_view_window
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical

import os
import numpy as np
import torch as t
import torch.nn as nn
import gym

from .utils import unwrap_time_limit, Smooth
from test.util_run_multi import *


class TestRunningStat(object):
    @pytest.mark.parametrize("shape", ((), (3,), (3, 4)))
    def test_push(self, shape):
        vals = []
        rs = RunningStat(shape)
        for _ in range(5):
            val = t.randn(shape, dtype=t.float64)
            rs.push(val)
            vals.append(val)
            m = t.mean(t.stack(vals), dim=0)
            assert t.allclose(rs.mean, m)
            v = (t.square(m)
                 if (len(vals) == 1)
                 else t.var(t.stack(vals), dim=0, unbiased=True))
            assert t.allclose(rs.var, v)

    @pytest.mark.parametrize("shape", ((), (3,), (3, 4)))
    def test_update(self, shape):
        rs1 = RunningStat(shape)
        rs2 = RunningStat(shape)
        rs = RunningStat(shape)
        for _ in range(5):
            val = t.randn(shape, dtype=t.float64)
            rs1.push(val)
            rs.push(val)
        for _ in range(9):
            val = t.randn(shape, dtype=t.float64)
            rs2.push(val)
            rs.push(val)
        rs1.update(rs2)
        assert t.allclose(rs.mean, rs1.mean)
        assert t.allclose(rs.std, rs1.std)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.tanh(
            self.fc3(
                self.fc2(
                    self.fc1(state)
                )
            )) * self.action_range
        return a


# class ActorDiscrete(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorDiscrete, self).__init__()
#
#         self.fc1 = nn.Linear(state_dim, 16)
#         self.fc2 = nn.Linear(16, action_dim)
#
#     def forward(self, state):
#         a = self.fc1(state)
#         a = Categorical(t.softmax(self.fc2(a), dim=1)).sample([1]).item()
#         return a

class ActorDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorDiscrete, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, action_dim)

    def forward(self, state):
        a = self.fc1(state)
        a = t.argmax(self.fc2(a), dim=1).item()
        return a

# class ActorDiscrete(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorDiscrete, self).__init__()
#         self.fc = nn.Linear(state_dim, action_dim, bias=False)
#
#     def forward(self, state):
#         a = t.argmax(self.fc(state), dim=1).item()
#         return a


class TestARS(object):
    # configs and definitions
    # disable_view_window()
    # c = Config()
    # c.env_name = "Pendulum-v0"
    # c.env = unwrap_time_limit(gym.make(c.env_name))
    # c.observe_dim = 3
    # c.action_dim = 1
    # c.action_range = 2
    # c.max_episodes = 100000
    # c.max_steps = 200
    # c.solved_reward = -150
    # c.solved_repeat = 5
    # c.device = "cpu"

    disable_view_window()
    c = Config()
    # Note: online policy algorithms such as PPO and A3C does not
    # work well in Pendulum (reason unknown)
    # and MountainCarContinuous (sparse returns)
    c.env_name = "CartPole-v0"
    c.env = unwrap_time_limit(gym.make(c.env_name))
    c.observe_dim = 4
    c.action_num = 2
    c.max_episodes = 100000
    c.max_steps = 40
    c.solved_reward = 30
    c.solved_repeat = 5
    c.device = "cpu"

    # @staticmethod
    # def ars():
    #     c = TestARS.c
    #     actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
    #                 .to(c.device), c.device, c.device)
    #     servers = model_server_helper(model_num=1)
    #     world = get_world()
    #     ars_group = world.create_rpc_group("ars", ["0", "1", "2"])
    #     ars = ARS(actor, t.optim.SGD, ars_group, servers,
    #               actor_learning_rate=0.1,
    #               noise_size=1000000,
    #               normalize_state=False)
    #     return ars

    @staticmethod
    def ars():
        c = TestARS.c
        actor = smw(ActorDiscrete(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        servers = model_server_helper(model_num=1)
        world = get_world()
        ars_group = world.create_rpc_group("ars", ["0", "1", "2"])
        ars = ARS(actor, t.optim.SGD, ars_group, servers,
                  actor_learning_rate=0.01,
                  noise_size=1000000,
                  rollout_num=32,
                  normalize_state=False)
        return ars

    @staticmethod
    def ars_lr():
        c = TestARS.c
        actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                    .to(c.device), c.device, c.device)
        lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)],
                                         logger=default_logger)
        servers = model_server_helper(model_num=1)
        world = get_world()
        ars_group = world.create_rpc_group("ars", ["0", "1", "2"])
        ars = ARS(actor, t.optim.SGD, ars_group, servers,
                  noise_size=1000000,
                  lr_scheduler=LambdaLR,
                  lr_scheduler_args=((lr_func,),))
        return ars

    ########################################################################
    # Test for ARS acting
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_act(_):
        ars = TestARS.ars()
        c = TestARS.c
        state = t.zeros([1, c.observe_dim])
        ars.act({"state": state}, "original")
        ars.act({"state": state}, ars.get_actor_types()[0])
        with pytest.raises(ValueError):
            ars.act({"state": state}, "some_invalid_actor_type")
        return True

    ########################################################################
    # Test for ARS storage
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_store_reward(_):
        ars = TestARS.ars()
        ars.store_reward(0.0, ars.get_actor_types()[0])
        with pytest.raises(ValueError):
            ars.store_reward(1.0, "some_invalid_actor_type")
        return True

    ########################################################################
    # Test for ARS update
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_update(_):
        ars = TestARS.ars()
        c = TestARS.c
        for at in ars.get_actor_types():
            # get action will cause filters to initialize
            _action = ars.act({"state": t.zeros([1, c.observe_dim])}, at)
            if at.startswith("neg"):
                ars.store_reward(1.0, at)
            else:
                ars.store_reward(0.0, at)
        ars.update()
        return True

    ########################################################################
    # Test for ARS save & load
    ########################################################################
    # Skipped, it is the same as base

    ########################################################################
    # Test for ARS lr_scheduler
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_lr_scheduler(_):
        ars = TestARS.ars_lr()
        ars.update_lr_scheduler()
        return True

    ########################################################################
    # Test for ARS full training.
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               timeout=18000)
    @WorldTestBase.setup_world
    def test_full_train(rank):
        c = TestARS.c
        ars = TestARS.ars()

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

            all_reward = 0
            for at in ars.get_actor_types():
                total_reward = 0

                # batch size = 1
                state = t.tensor(env.reset(), dtype=t.float32, device=c.device)
                while not terminal and step <= c.max_steps:
                    step.count()
                    with t.no_grad():
                        old_state = state
                        # agent model inference
                        action = ars.act({"state": old_state.unsqueeze(0)}, at)
                        _, reward, __, ___ = env.step(action)
                        total_reward += float(reward)
                step.reset()
                terminal = False
                ars.store_reward(total_reward, at)
                all_reward += total_reward

            # update
            # default_logger.critical("Process {} rs:{}".format(
            #     rank, ars.filter
            # ))
            ars.update()
            smoother.update(all_reward / len(ars.get_actor_types()))
            default_logger.info("Process {} Episode {} total reward={:.2f}"
                                .format(rank, episode, smoother.value))

            if smoother.value > c.solved_reward:
                reward_fulfilled.count()
                if reward_fulfilled >= c.solved_repeat:
                    default_logger.info("Environment solved!")
                    return True
            else:
                reward_fulfilled.reset()

        raise RuntimeError("ARS Training failed.")
