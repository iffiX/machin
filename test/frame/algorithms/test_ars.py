from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.ars import ARS, RunningStat
from machin.frame.helpers.servers import model_server_helper
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.utils.learning_rate import gen_learning_rate_func
from machin.env.utils.openai_gym import disable_view_window
from torch.optim.lr_scheduler import LambdaLR

import os
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


# class ActorDiscrete(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorDiscrete, self).__init__()
#
#         self.fc1 = nn.Linear(state_dim, 16)
#         self.fc2 = nn.Linear(16, action_dim)
#
#     def forward(self, state):
#         a = self.fc1(state)
#         a = t.argmax(self.fc2(a), dim=1).item()
#         return a

class ActorDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorDiscrete, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim, bias=False)

    def forward(self, state):
        a = t.argmax(self.fc(state), dim=1).item()
        return a


class TestARS(object):
    # configs and definitions
    # Cartpole-v0 can be solved:
    # within 200 episodes, using single layer Actor
    # within 400 episodes, using double layer Actor

    # However, ARS fail to deal with pendulum v0:
    # Actor((st, 16)->(16, a)), noise_std=0.01, lr=0.05, rollout=9, optim=Adam)
    # reaches mean score = -700 at 10000 episodes
    # Actor((st, a)), noise_std=0.01, lr=0.05, rollout=9, optim=Adam)
    # reaches mean score = -1100 at 15000 episodes
    # and Adam optimizer is better than SGD
    disable_view_window()
    c = Config()
    c.env_name = "CartPole-v0"
    c.env = unwrap_time_limit(gym.make(c.env_name))
    c.observe_dim = 4
    c.action_num = 2
    c.max_episodes = 1000
    c.max_steps = 200
    c.solved_reward = 190
    c.solved_repeat = 5

    @staticmethod
    def ars():
        c = TestARS.c
        actor = smw(ActorDiscrete(c.observe_dim, c.action_num)
                    .to(c.device), c.device, c.device)
        servers = model_server_helper(model_num=1)
        world = get_world()
        ars_group = world.create_rpc_group("ars", ["0", "1", "2"])
        ars = ARS(actor, t.optim.SGD, ars_group, servers,
                  noise_std_dev=0.1,
                  learning_rate=0.1,
                  noise_size=1000000,
                  rollout_num=6,
                  used_rollout_num=6,
                  normalize_state=True)
        return ars

    @staticmethod
    def ars_lr():
        c = TestARS.c
        actor = smw(ActorDiscrete(c.observe_dim, c.action_num)
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
    @run_multi(expected_results=[True, True, True],
               pass_through=["pytestconfig"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_act(_, pytestconfig):
        c = TestARS.c
        c.device = pytestconfig.get_option("gpu_device")
        ars = TestARS.ars()
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
    @run_multi(expected_results=[True, True, True],
               pass_through=["pytestconfig"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_store_reward(_, pytestconfig):
        c = TestARS.c
        c.device = pytestconfig.get_option("gpu_device")
        ars = TestARS.ars()
        ars.store_reward(0.0, ars.get_actor_types()[0])
        with pytest.raises(ValueError):
            ars.store_reward(1.0, "some_invalid_actor_type")
        return True

    ########################################################################
    # Test for ARS update
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["pytestconfig"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_update(_, pytestconfig):
        c = TestARS.c
        c.device = pytestconfig.get_option("gpu_device")
        ars = TestARS.ars()
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
    @run_multi(expected_results=[True, True, True],
               pass_through=["pytestconfig"],
               timeout=180)
    @WorldTestBase.setup_world
    def test_lr_scheduler(_, pytestconfig):
        c = TestARS.c
        c.device = pytestconfig.get_option("gpu_device")
        ars = TestARS.ars_lr()
        ars.update_lr_scheduler()
        return True

    ########################################################################
    # Test for ARS full training.
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True],
               pass_through=["pytestconfig"],
               timeout=1800)
    @WorldTestBase.setup_world
    def test_full_train(rank, pytestconfig):
        c = TestARS.c
        c.device = pytestconfig.get_option("gpu_device")
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
                        # agent model inference
                        action = ars.act({"state": state.unsqueeze(0)}, at)
                        state, reward, terminal, __ = env.step(action)
                        state = t.tensor(state, dtype=t.float32,
                                         device=c.device)
                        total_reward += float(reward)
                step.reset()
                terminal = False
                ars.store_reward(total_reward, at)
                all_reward += total_reward

            # update
            ars.update()
            smoother.update(all_reward / len(ars.get_actor_types()))
            default_logger.info("Process {} Episode {} total reward={:.2f}"
                                .format(rank, episode, smoother.value))

            if smoother.value > c.solved_reward:
                reward_fulfilled.count()
                if reward_fulfilled >= c.solved_repeat:
                    default_logger.info("Environment solved!")
                    raise SafeExit
            else:
                reward_fulfilled.reset()

        raise RuntimeError("ARS Training failed.")
