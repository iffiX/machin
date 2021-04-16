from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.apex import DQNApex, DDPGApex
from machin.frame.helpers.servers import model_server_helper
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window

import os
import torch as t
import torch.nn as nn
import gym

from test.frame.algorithms.utils import unwrap_time_limit, Smooth
from test.util_run_multi import *
from test.util_fixtures import *


class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a


class ActorDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.softmax(self.fc3(a), dim=1)
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TestDQNApex:
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
    c.max_episodes = 2000
    c.max_steps = 200
    c.replay_size = 100000
    c.solved_reward = 150
    c.solved_repeat = 5

    @staticmethod
    def dqn_apex(device, dtype):
        c = TestDQNApex.c
        q_net = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        q_net_t = smw(
            QNet(c.observe_dim, c.action_num).type(dtype).to(device), device, device
        )
        servers = model_server_helper(model_num=1)
        world = get_world()
        # process 0 and 1 will be workers, and 2 will be trainer
        apex_group = world.create_rpc_group("apex", ["0", "1", "2"])
        dqn_apex = DQNApex(
            q_net,
            q_net_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            apex_group,
            servers,
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        return dqn_apex

    ########################################################################
    # Test for DQNApex acting
    ########################################################################
    @staticmethod
    @run_multi(
        expected_results=[True, True, True],
        pass_through=["device", "dtype"],
        timeout=180,
    )
    @WorldTestBase.setup_world
    def test_act(_, device, dtype):
        c = TestDQNApex.c
        dqn_apex = TestDQNApex.dqn_apex(device, dtype)
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        dqn_apex.act_discrete({"state": state})
        dqn_apex.act_discrete({"state": state}, True)
        dqn_apex.act_discrete_with_noise({"state": state})
        dqn_apex.act_discrete_with_noise({"state": state}, True)
        return True

    ########################################################################
    # Test for DQNApex criticizing
    ########################################################################
    @staticmethod
    @run_multi(
        expected_results=[True, True, True],
        pass_through=["device", "dtype"],
        timeout=180,
    )
    @WorldTestBase.setup_world
    def test_criticize(_, device, dtype):
        c = TestDQNApex.c
        dqn_apex = TestDQNApex.dqn_apex(device, dtype)
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        dqn_apex._criticize({"state": state})
        dqn_apex._criticize({"state": state}, True)
        return True

    ########################################################################
    # Test for DQNApex storage
    ########################################################################
    # Skipped, it is the same as DQN

    ########################################################################
    # Test for DQNApex update
    ########################################################################
    @staticmethod
    @run_multi(
        expected_results=[True, True, True],
        pass_through=["device", "dtype"],
        timeout=180,
    )
    @WorldTestBase.setup_world
    def test_update(rank, device, dtype):
        c = TestDQNApex.c
        dqn_apex = TestDQNApex.dqn_apex(device, dtype)
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, 1], dtype=t.int)
        if rank in (0, 1):
            dqn_apex.store_episode(
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
            dqn_apex.manual_sync()
        if rank == 2:
            sleep(2)
            dqn_apex.update()
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
    # Test for DQNApex config & init
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True], timeout=180)
    @WorldTestBase.setup_world
    def test_config_init(rank):
        c = TestDQNApex.c
        config = DQNApex.generate_config({})
        config["frame_config"]["models"] = ["QNet", "QNet"]
        config["frame_config"]["model_kwargs"] = [
            {"state_dim": c.observe_dim, "action_num": c.action_num}
        ] * 2
        dqn_apex = DQNApex.init_from_config(config)

        old_state = state = t.zeros([1, c.observe_dim], dtype=t.float32)
        action = t.zeros([1, 1], dtype=t.int)
        if rank in (1, 2):
            dqn_apex.store_episode(
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
            dqn_apex.manual_sync()
        if rank == 0:
            sleep(2)
            dqn_apex.update(
                update_value=True, update_target=True, concatenate_samples=True
            )
        return True

    ########################################################################
    # Test for DQNApex full training.
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True], timeout=1800)
    @WorldTestBase.setup_world
    def test_full_train(rank):
        c = TestDQNApex.c
        dqn_apex = TestDQNApex.dqn_apex("cpu", t.float32)
        # perform manual syncing to decrease the number of rpc calls
        dqn_apex.set_sync(False)

        # begin training
        episode, step = Counter(), Counter()
        reward_fulfilled = Counter()
        smoother = Smooth()
        terminal = False

        env = c.env
        world = get_world()
        all_group = world.create_rpc_group("all", ["0", "1", "2"])
        all_group.pair(f"{rank}_running", True)

        if rank in (0, 1):
            while episode < c.max_episodes:
                # wait for trainer to keep up
                sleep(0.2)
                episode.count()

                # batch size = 1
                total_reward = 0
                state = t.tensor(env.reset(), dtype=t.float32)

                dqn_apex.manual_sync()
                while not terminal and step <= c.max_steps:
                    step.count()
                    with t.no_grad():
                        old_state = state
                        # agent model inference
                        action = dqn_apex.act_discrete_with_noise(
                            {"state": old_state.unsqueeze(0)}
                        )
                        state, reward, terminal, _ = env.step(action.item())
                        state = t.tensor(state, dtype=t.float32).flatten()
                        total_reward += float(reward)

                        dqn_apex.store_transition(
                            {
                                "state": {"state": old_state.unsqueeze(0)},
                                "action": {"action": action},
                                "next_state": {"state": state.unsqueeze(0)},
                                "reward": float(reward),
                                "terminal": terminal or step == c.max_steps,
                            }
                        )
                smoother.update(total_reward)
                step.reset()
                terminal = False

                default_logger.info(
                    "Process {} Episode {} total reward={:.2f}".format(
                        rank, episode, smoother.value
                    )
                )

                if smoother.value > c.solved_reward:
                    reward_fulfilled.count()
                    if reward_fulfilled >= c.solved_repeat:
                        default_logger.info("Environment solved!")

                        all_group.unpair(f"{rank}_running")
                        while all_group.is_paired("0_running") or all_group.is_paired(
                            "1_running"
                        ):
                            # wait for all workers to join
                            sleep(1)
                        # wait for trainer
                        sleep(5)
                        return True
                else:
                    reward_fulfilled.reset()
        else:
            # wait for some samples
            while dqn_apex.replay_buffer.all_size() < 500:
                sleep(0.1)
            while all_group.is_paired("0_running") or all_group.is_paired("1_running"):
                dqn_apex.update()
                default_logger.info("Updated")
            return True

        raise RuntimeError("DQN-Apex Training failed.")


class TestDDPGApex:
    # configs and definitions
    disable_view_window()
    c = Config()
    c.env_name = "Pendulum-v0"
    c.env = unwrap_time_limit(gym.make(c.env_name))
    c.observe_dim = 3
    c.action_dim = 1
    c.action_range = 2
    c.max_episodes = 2000
    c.max_steps = 200
    c.noise_param = (0, 0.2)
    c.noise_mode = "normal"
    c.replay_size = 100000
    c.noise_interval = 2
    # takes too much computing resource
    # decrease standard for faster validation
    c.solved_reward = -400
    c.solved_repeat = 5

    @staticmethod
    def ddpg_apex(device, dtype, discrete=False):
        c = TestDDPGApex.c
        if not discrete:
            actor = smw(
                Actor(c.observe_dim, c.action_dim, c.action_range)
                .type(dtype)
                .to(device),
                device,
                device,
            )
            actor_t = smw(
                Actor(c.observe_dim, c.action_dim, c.action_range)
                .type(dtype)
                .to(device),
                device,
                device,
            )
        else:
            actor = smw(
                ActorDiscrete(c.observe_dim, c.action_dim).type(dtype).to(device),
                device,
                device,
            )
            actor_t = smw(
                ActorDiscrete(c.observe_dim, c.action_dim).type(dtype).to(device),
                device,
                device,
            )
        critic = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )

        servers = model_server_helper(model_num=2)
        world = get_world()
        # process 0 and 1 will be workers, and 2 will be trainer
        apex_group = world.create_rpc_group("worker", ["0", "1", "2"])
        ddpg_apex = DDPGApex(
            actor,
            actor_t,
            critic,
            critic_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            apex_group,
            servers,
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        return ddpg_apex

    ########################################################################
    # Test for DDPGApex contiguous domain acting
    ########################################################################
    @staticmethod
    @run_multi(
        expected_results=[True, True, True],
        pass_through=["device", "dtype"],
        timeout=180,
    )
    @WorldTestBase.setup_world
    def test_contiguous_act(_, device, dtype):
        c = TestDDPGApex.c
        ddpg_apex = TestDDPGApex.ddpg_apex(device, dtype)
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        ddpg_apex.act({"state": state})
        ddpg_apex.act({"state": state}, use_target=True)
        ddpg_apex.act_with_noise({"state": state}, noise_param=(0, 1.0), mode="uniform")
        ddpg_apex.act_with_noise(
            {"state": state}, noise_param=(0, 1.0), mode="uniform", use_target=True
        )
        return True

    ########################################################################
    # Test for DDPGApex discrete domain acting
    ########################################################################
    @staticmethod
    @run_multi(
        expected_results=[True, True, True],
        pass_through=["device", "dtype"],
        timeout=180,
    )
    @WorldTestBase.setup_world
    def test_discrete_act(_, device, dtype):
        c = TestDDPGApex.c
        c.device = gpu
        ddpg_apex = TestDDPGApex.ddpg_apex(device, dtype, discrete=True)
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        ddpg_apex.act_discrete({"state": state})
        ddpg_apex.act_discrete({"state": state}, use_target=True)
        ddpg_apex.act_discrete_with_noise({"state": state})
        ddpg_apex.act_discrete_with_noise({"state": state}, use_target=True)
        return True

    ########################################################################
    # Test for DDPGApex criticizing
    ########################################################################
    @staticmethod
    @run_multi(
        expected_results=[True, True, True],
        pass_through=["device", "dtype"],
        timeout=180,
    )
    @WorldTestBase.setup_world
    def test__criticize(_, device, dtype):
        c = TestDDPGApex.c
        c.device = gpu
        ddpg_apex = TestDDPGApex.ddpg_apex(device, dtype)
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, c.action_dim], dtype=dtype)
        ddpg_apex._criticize({"state": state}, {"action": action})
        ddpg_apex._criticize({"state": state}, {"action": action}, use_target=True)
        return True

    ########################################################################
    # Test for DDPGApex storage
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for DDPGApex update
    ########################################################################
    @staticmethod
    @run_multi(
        expected_results=[True, True, True],
        pass_through=["device", "dtype"],
        timeout=180,
    )
    @WorldTestBase.setup_world
    def test_update(rank, device, dtype):
        c = TestDDPGApex.c
        c.device = gpu
        ddpg_apex = TestDDPGApex.ddpg_apex(device, dtype)
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, c.action_dim], dtype=dtype)
        if rank in (0, 1):
            ddpg_apex.store_transition(
                {
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": 0,
                    "terminal": False,
                }
            )
            sleep(5)
            ddpg_apex.manual_sync()
        if rank == 2:
            sleep(2)
            ddpg_apex.update(
                update_value=True,
                update_policy=True,
                update_target=True,
                concatenate_samples=True,
            )
        return True

    ########################################################################
    # Test for DDPGApex save & load
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for DDPGApex lr_scheduler
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for DDPGApex config & init
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True], timeout=180)
    @WorldTestBase.setup_world
    def test_config_init(rank):
        c = TestDDPGApex.c
        config = DDPGApex.generate_config({})
        config["frame_config"]["models"] = ["Actor", "Actor", "Critic", "Critic"]
        config["frame_config"]["model_kwargs"] = [
            {
                "state_dim": c.observe_dim,
                "action_dim": c.action_dim,
                "action_range": c.action_range,
            }
        ] * 2 + [{"state_dim": c.observe_dim, "action_dim": c.action_dim}] * 2
        ddpg_apex = DDPGApex.init_from_config(config)

        old_state = state = t.zeros([1, c.observe_dim], dtype=t.float32)
        action = t.zeros([1, c.action_dim], dtype=t.float32)
        if rank in (1, 2):
            ddpg_apex.store_transition(
                {
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": 0,
                    "terminal": False,
                }
            )
            sleep(5)
            ddpg_apex.manual_sync()
        if rank == 0:
            sleep(2)
            ddpg_apex.update()

        return True

    ########################################################################
    # Test for DDPGApex full training.
    ########################################################################
    @staticmethod
    @run_multi(expected_results=[True, True, True], timeout=1800)
    @WorldTestBase.setup_world
    def test_full_train(rank):
        # from logging import DEBUG
        # from machin.utils.logging import default_logger as df
        #
        # df.setLevel(DEBUG)

        c = TestDDPGApex.c
        ddpg_apex = TestDDPGApex.ddpg_apex("cpu", t.float32)
        # perform manual syncing to decrease the number of rpc calls
        ddpg_apex.set_sync(False)

        # begin training
        episode, step = Counter(), Counter()
        avg_step = Smooth()
        reward_fulfilled = Counter()
        smoother = Smooth()
        terminal = False

        env = c.env
        world = get_world()
        all_group = world.create_rpc_group("all", ["0", "1", "2"])
        all_group.pair(f"{rank}_running", True)
        all_group.pair(f"{rank}_step", avg_step)
        default_logger.info(f"{rank}, pid {os.getpid()}")
        if rank == 0:
            all_group.pair("episode", episode)

        if rank in (0, 1):
            while episode < c.max_episodes:
                # wait for trainer to keep up
                sleep(1)
                episode.count()

                # batch size = 1
                total_reward = 0
                state = t.tensor(env.reset(), dtype=t.float32)

                ddpg_apex.manual_sync()
                while not terminal and step <= c.max_steps:
                    step.count()
                    with t.no_grad():
                        old_state = state
                        if episode.get() % c.noise_interval == 0:
                            action = ddpg_apex.act_with_noise(
                                {"state": old_state.unsqueeze(0)},
                                noise_param=c.noise_param,
                                mode=c.noise_mode,
                            )
                        else:
                            action = ddpg_apex.act(
                                {"state": old_state.unsqueeze(0)}
                            ).clamp(-c.action_range, c.action_range)

                        state, reward, terminal, _ = env.step(action.cpu().numpy())
                        state = t.tensor(state, dtype=t.float32).flatten()
                        total_reward += float(reward)

                        ddpg_apex.store_transition(
                            {
                                "state": {"state": old_state.unsqueeze(0)},
                                "action": {"action": action},
                                "next_state": {"state": state.unsqueeze(0)},
                                "reward": float(reward),
                                "terminal": terminal or step == c.max_steps,
                            }
                        )

                smoother.update(total_reward)
                avg_step.update(step.get())
                step.reset()
                terminal = False

                if episode.get() % c.noise_interval != 0:
                    # only log result without noise
                    default_logger.info(
                        "Process {} Episode {} "
                        "total reward={:.2f}".format(rank, episode, smoother.value)
                    )

                if smoother.value > c.solved_reward:
                    reward_fulfilled.count()
                    if reward_fulfilled >= c.solved_repeat:
                        default_logger.info("Environment solved!")

                        all_group.unpair(f"{rank}_running")
                        while all_group.is_paired("0_running") or all_group.is_paired(
                            "1_running"
                        ):
                            # wait for all workers to join
                            sleep(1)
                        # wait for trainer
                        sleep(5)
                        return True
                else:
                    reward_fulfilled.reset()
        else:
            # wait for some samples
            while ddpg_apex.replay_buffer.all_size() < 500:
                sleep(0.1)
            while all_group.is_paired("0_running") or all_group.is_paired("1_running"):
                p0_step = all_group.get_paired("0_step").to_here().value
                p1_step = all_group.get_paired("1_step").to_here().value
                # not accurate since sampler processes are not synchronized, but works.
                update_times = int((p0_step + p1_step) / 2)
                if update_times == 0:
                    continue
                for i in range(update_times):
                    ddpg_apex.update()
                default_logger.info(f"Update {update_times} times")
            return True

        raise RuntimeError("DDPG-Apex Training failed.")
