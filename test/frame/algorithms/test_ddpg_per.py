from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.ddpg_per import DDPGPer
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window

import pytest
import torch as t
import torch.nn as nn
import gym

from .utils import unwrap_time_limit, Smooth


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TestDDPGPer(object):
    # configs and definitions
    @pytest.fixture(scope="class")
    def train_config(self, pytestconfig):
        disable_view_window()
        c = Config()
        c.env_name = "Pendulum-v0"
        c.env = unwrap_time_limit(gym.make(c.env_name))
        c.observe_dim = 3
        c.action_dim = 1
        c.action_range = 2
        c.max_episodes = 1000
        c.max_steps = 200
        c.noise_param = (0, 0.2)
        c.noise_mode = "normal"
        c.noise_interval = 2
        c.replay_size = 100000
        c.solved_reward = -150
        c.solved_repeat = 5
        c.device = pytestconfig.get_option("gpu_device")
        return c

    @pytest.fixture(scope="function")
    def ddpg_per(self, train_config):
        c = train_config
        actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                    .to(c.device), c.device, c.device)
        actor_t = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                      .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim, c.action_dim)
                     .to(c.device), c.device, c.device)
        critic_t = smw(Critic(c.observe_dim, c.action_dim)
                       .to(c.device), c.device, c.device)
        ddpg_per = DDPGPer(actor, actor_t, critic, critic_t,
                           t.optim.Adam,
                           nn.MSELoss(reduction='sum'),
                           replay_device="cpu",
                           replay_size=c.replay_size)
        return ddpg_per

    @pytest.fixture(scope="function")
    def ddpg_per_vis(self, train_config, tmpdir):
        # not used for training, only used for testing apis
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                    .to(c.device), c.device, c.device)
        actor_t = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                      .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim, c.action_dim)
                     .to(c.device), c.device, c.device)
        critic_t = smw(Critic(c.observe_dim, c.action_dim)
                       .to(c.device), c.device, c.device)
        ddpg_per = DDPGPer(actor, actor_t, critic, critic_t,
                           t.optim.Adam,
                           nn.MSELoss(reduction='sum'),
                           replay_device="cpu",
                           replay_size=c.replay_size,
                           visualize=True,
                           visualize_dir=str(tmp_dir))
        return ddpg_per

    ########################################################################
    # Test for DDPGPer criterion (mainly code coverage)
    ########################################################################
    def test_criterion(self, train_config):
        c = train_config
        actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                    .to(c.device), c.device, c.device)
        actor_t = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                      .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim, c.action_dim)
                     .to(c.device), c.device, c.device)
        critic_t = smw(Critic(c.observe_dim, c.action_dim)
                       .to(c.device), c.device, c.device)
        with pytest.raises(RuntimeError,
                           match="Criterion does not have the "
                                 "'reduction' property"):
            def criterion(a, b):
                return a - b

            _ = DDPGPer(actor, actor_t, critic, critic_t,
                        t.optim.Adam,
                        criterion,
                        replay_device="cpu",
                        replay_size=c.replay_size)

    ########################################################################
    # Test for DDPGPer contiguous domain acting
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for DDPGPer discrete domain acting
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for DDPGPer criticizing
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for DDPGPer storage
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for DDPGPer update
    ########################################################################
    def test_update(self, train_config, ddpg_per_vis):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, c.action_dim])
        ddpg_per_vis.store_transition({
            "state": {"state": old_state},
            "action": {"action": action},
            "next_state": {"state": state},
            "reward": 0,
            "terminal": False
        })
        ddpg_per_vis.update(update_value=True, update_policy=True,
                            update_target=True, concatenate_samples=True)
        ddpg_per_vis.update(update_value=False, update_policy=False,
                            update_target=False, concatenate_samples=True)

    ########################################################################
    # Test for DDPGPer save & load
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for DDPGPer lr_scheduler
    ########################################################################
    # Skipped, it is the same as DDPG

    ########################################################################
    # Test for DDPGPer full training.
    ########################################################################
    def test_full_train(self, train_config, ddpg_per):
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
                    if episode.get() % c.noise_interval == 0:
                        action = ddpg_per.act_with_noise(
                            {"state": old_state.unsqueeze(0)},
                            noise_param=c.noise_param,
                            mode=c.noise_mode
                        )
                    else:
                        action = ddpg_per.act(
                            {"state": old_state.unsqueeze(0)}
                        ).clamp(-c.action_range, c.action_range)

                    state, reward, terminal, _ = env.step(action.cpu().numpy())
                    state = t.tensor(state, dtype=t.float32, device=c.device) \
                        .flatten()
                    total_reward += float(reward)

                    ddpg_per.store_transition({
                        "state": {"state": old_state.unsqueeze(0)},
                        "action": {"action": action},
                        "next_state": {"state": state.unsqueeze(0)},
                        "reward": float(reward),
                        "terminal": terminal or step == c.max_steps
                    })
            # update
            if episode > 100:
                for i in range(step.get()):
                    ddpg_per.update()

            smoother.update(total_reward)
            step.reset()
            terminal = False

            if episode.get() % c.noise_interval != 0:
                # only log result without noise
                logger.info("Episode {} total reward={:.2f}"
                            .format(episode, smoother.value))

            if smoother.value > c.solved_reward:
                reward_fulfilled.count()
                if reward_fulfilled >= c.solved_repeat:
                    logger.info("Environment solved!")
                    return
            else:
                reward_fulfilled.reset()

        pytest.fail("DDPGPer Training failed.")
