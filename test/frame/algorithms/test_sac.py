from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import softplus
from torch.distributions import Normal
from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.sac import SAC
from machin.utils.learning_rate import gen_learning_rate_func
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
from test.frame.algorithms.utils import unwrap_time_limit, Smooth
from test.util_fixtures import *
from test.util_platforms import linux_only

import pytest
import torch as t
import torch.nn as nn
import gym


def atanh(x):
    return 0.5 * t.log((1 + x) / (1 - x))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.mu_head = nn.Linear(16, action_dim)
        self.sigma_head = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        mu = self.mu_head(a)
        sigma = softplus(self.sigma_head(a))
        dist = Normal(mu, sigma)
        act = (
            atanh(action / self.action_range) if action is not None else dist.rsample()
        )
        act_entropy = dist.entropy()

        # the suggested way to confine your actions within a valid range
        # is not clamping, but remapping the distribution
        act_log_prob = dist.log_prob(act)
        act_tanh = t.tanh(act)
        act = act_tanh * self.action_range

        # the distribution remapping process used in the original essay.
        act_log_prob -= t.log(self.action_range * (1 - act_tanh.pow(2)) + 1e-6)
        act_log_prob = act_log_prob.sum(1, keepdim=True)

        # If your distribution is different from "Normal" then you may either:
        # 1. deduce the remapping function for your distribution and clamping
        #    function such as tanh
        # 2. clamp you action, but please take care:
        #    1. do not clamp actions before calculating their log probability,
        #       because the log probability of clamped actions might will be
        #       extremely small, and will cause nan
        #    2. do not clamp actions after sampling and before storing them in
        #       the replay buffer, because during update, log probability will
        #       be re-evaluated they might also be extremely small, and network
        #       will "nan". (might happen in PPO, not in SAC because there is
        #       no re-evaluation)
        # Only clamp actions sent to the environment, this is equivalent to
        # change the action reward distribution, will not cause "nan", but
        # this makes your training environment further differ from you real
        # environment.
        return act, act_log_prob, act_entropy


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


class TestSAC:
    # configs and definitions
    @pytest.fixture(scope="class")
    def train_config(self):
        disable_view_window()
        c = Config()
        c.env_name = "Pendulum-v0"
        c.env = unwrap_time_limit(gym.make(c.env_name))
        c.observe_dim = 3
        c.action_dim = 1
        c.action_range = 2
        c.max_episodes = 1000
        c.max_steps = 200
        c.replay_size = 100000
        c.solved_reward = -400
        c.solved_repeat = 5
        return c

    @pytest.fixture(scope="function")
    def sac(self, train_config, device, dtype):
        c = train_config
        actor = smw(
            Actor(c.observe_dim, c.action_dim, c.action_range).type(dtype).to(device),
            device,
            device,
        )
        critic = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2 = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        sac = SAC(
            actor,
            critic,
            critic_t,
            critic2,
            critic2_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        return sac

    @pytest.fixture(scope="function")
    def sac_vis(self, train_config, device, dtype, tmpdir):
        # not used for training, only used for testing apis
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        actor = smw(
            Actor(c.observe_dim, c.action_dim, c.action_range).type(dtype).to(device),
            device,
            device,
        )
        critic = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2 = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        sac = SAC(
            actor,
            critic,
            critic_t,
            critic2,
            critic2_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
            visualize=True,
            visualize_dir=str(tmp_dir),
        )
        return sac

    @pytest.fixture(scope="function")
    def sac_lr(self, train_config, device, dtype):
        # not used for training, only used for testing apis
        c = train_config
        actor = smw(
            Actor(c.observe_dim, c.action_dim, c.action_range).type(dtype).to(device),
            device,
            device,
        )
        critic = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2 = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        critic2_t = smw(
            Critic(c.observe_dim, c.action_dim).type(dtype).to(device), device, device
        )
        lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)], logger=logger)
        with pytest.raises(TypeError, match="missing .+ positional argument"):
            _ = SAC(
                actor,
                critic,
                critic_t,
                critic2,
                critic2_t,
                t.optim.Adam,
                nn.MSELoss(reduction="sum"),
                replay_device="cpu",
                replay_size=c.replay_size,
                lr_scheduler=LambdaLR,
            )
        sac = SAC(
            actor,
            critic,
            critic_t,
            critic2,
            critic2_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
            lr_scheduler=LambdaLR,
            lr_scheduler_args=((lr_func,), (lr_func,), (lr_func,)),
        )
        return sac

    @pytest.fixture(scope="function")
    def sac_train(self, train_config):
        c = train_config
        actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range), "cpu", "cpu")
        critic = smw(Critic(c.observe_dim, c.action_dim), "cpu", "cpu")
        critic_t = smw(Critic(c.observe_dim, c.action_dim), "cpu", "cpu")
        critic2 = smw(Critic(c.observe_dim, c.action_dim), "cpu", "cpu")
        critic2_t = smw(Critic(c.observe_dim, c.action_dim), "cpu", "cpu")
        sac = SAC(
            actor,
            critic,
            critic_t,
            critic2,
            critic2_t,
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            replay_device="cpu",
            replay_size=c.replay_size,
        )
        return sac

    ########################################################################
    # Test for SAC acting
    ########################################################################
    def test_act(self, train_config, sac, dtype):
        c = train_config
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        sac.act({"state": state})

    ########################################################################
    # Test for SAC criticizing
    ########################################################################
    def test__criticize(self, train_config, sac, dtype):
        c = train_config
        state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, c.action_dim], dtype=dtype)
        sac._criticize({"state": state}, {"action": action})
        sac._criticize({"state": state}, {"action": action}, use_target=True)
        sac._criticize2({"state": state}, {"action": action})
        sac._criticize2({"state": state}, {"action": action}, use_target=True)

    ########################################################################
    # Test for SAC storage
    ########################################################################
    def test_store(self, train_config, sac, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, c.action_dim], dtype=dtype)
        sac.store_episode(
            [
                {
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": 0,
                    "terminal": False,
                }
            ]
        )

    ########################################################################
    # Test for SAC update
    ########################################################################
    def test_update(self, train_config, sac_vis, dtype):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim], dtype=dtype)
        action = t.zeros([1, c.action_dim], dtype=dtype)
        sac_vis.store_episode(
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
        # heuristic entropy
        sac_vis.target_entropy = -c.action_dim
        sac_vis.update(
            update_value=True,
            update_policy=True,
            update_target=True,
            update_entropy_alpha=True,
            concatenate_samples=True,
        )
        sac_vis.update(
            update_value=False,
            update_policy=False,
            update_target=False,
            update_entropy_alpha=False,
            concatenate_samples=True,
        )

    ########################################################################
    # Test for SAC save & load
    ########################################################################
    def test_save_load(self, train_config, sac, tmpdir):
        save_dir = tmpdir.make_numbered_dir()
        sac.save(
            model_dir=str(save_dir),
            network_map={
                "critic_target": "critic_t",
                "critic2_target": "critic2_t",
                "actor": "actor",
            },
            version=1000,
        )
        sac.load(
            model_dir=str(save_dir),
            network_map={
                "critic_target": "critic_t",
                "critic2_target": "critic2_t",
                "actor": "actor",
            },
            version=1000,
        )

    ########################################################################
    # Test for SAC lr_scheduler
    ########################################################################
    def test_lr_scheduler(self, train_config, sac_lr):
        sac_lr.update_lr_scheduler()

    ########################################################################
    # Test for SAC config & init
    ########################################################################
    def test_config_init(self, train_config):
        c = train_config
        config = SAC.generate_config({})
        config["frame_config"]["models"] = [
            "Actor",
            "Critic",
            "Critic",
            "Critic",
            "Critic",
        ]
        config["frame_config"]["model_kwargs"] = [
            {
                "state_dim": c.observe_dim,
                "action_dim": c.action_dim,
                "action_range": c.action_range,
            }
        ] + [{"state_dim": c.observe_dim, "action_dim": c.action_dim}] * 4
        sac = SAC.init_from_config(config)

        old_state = state = t.zeros([1, c.observe_dim], dtype=t.float32)
        action = t.zeros([1, c.action_dim], dtype=t.float32)
        sac.store_episode(
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
        # heuristic entropy
        sac.target_entropy = -c.action_dim
        sac.update()

    ########################################################################
    # Test for SAC full training.
    ########################################################################
    @linux_only
    def test_full_train(self, train_config, sac_train):
        c = train_config
        sac_train.target_entropy = -c.action_dim

        # begin training
        episode, step = Counter(), Counter()
        reward_fulfilled = Counter()
        smoother = Smooth()
        terminal = False

        env = c.env
        env.seed(0)
        while episode < c.max_episodes:
            episode.count()

            # batch size = 1
            total_reward = 0
            state = t.tensor(env.reset(), dtype=t.float32)
            tmp_observations = []

            while not terminal and step <= c.max_steps:
                step.count()
                with t.no_grad():
                    old_state = state

                    # agent model inference
                    action = sac_train.act({"state": old_state.unsqueeze(0)})[0]

                    state, reward, terminal, _ = env.step(action.cpu().numpy())
                    state = t.tensor(state, dtype=t.float32).flatten()
                    total_reward += float(reward)

                    tmp_observations.append(
                        {
                            "state": {"state": old_state.unsqueeze(0)},
                            "action": {"action": action},
                            "next_state": {"state": state.unsqueeze(0)},
                            "reward": float(reward),
                            "terminal": terminal or step == c.max_steps,
                        }
                    )

            sac_train.store_episode(tmp_observations)
            # update
            if episode > 100:
                for i in range(step.get()):
                    sac_train.update()
                logger.info(f"new entropy alpha: {sac_train.entropy_alpha.item()}")

            smoother.update(total_reward)
            step.reset()
            terminal = False

            logger.info(f"Episode {episode} total reward={smoother.value:.2f}")

            if smoother.value > c.solved_reward:
                reward_fulfilled.count()
                if reward_fulfilled >= c.solved_repeat:
                    logger.info("Environment solved!")
                    return
            else:
                reward_fulfilled.reset()

        pytest.fail("SAC Training failed.")
