from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.ddpg import DDPG
from machin.utils.learning_rate import gen_learning_rate_func
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
from torch.optim.lr_scheduler import LambdaLR

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


class ActorDiscreet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorDiscreet, self).__init__()

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


class TestDDPG(object):
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
        c.replay_size = 10000
        c.solved_reward = -150
        c.solved_repeat = 5
        c.device = "cpu"
        return c

    @pytest.fixture(scope="function")
    def ddpg(self, train_config):
        c = train_config
        actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                    .to(c.device), c.device, c.device)
        actor_t = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                      .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim, c.action_dim)
                     .to(c.device), c.device, c.device)
        critic_t = smw(Critic(c.observe_dim, c.action_dim)
                       .to(c.device), c.device, c.device)
        ddpg = DDPG(actor, actor_t, critic, critic_t,
                    t.optim.Adam,
                    nn.MSELoss(reduction='sum'),
                    replay_device=c.device,
                    replay_size=c.replay_size)
        return ddpg

    @pytest.fixture(scope="function")
    def ddpg_vis(self, train_config, tmpdir):
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
        ddpg = DDPG(actor, actor_t, critic, critic_t,
                    t.optim.Adam,
                    nn.MSELoss(reduction='sum'),
                    replay_device=c.device,
                    replay_size=c.replay_size,
                    visualize=True,
                    visualize_dir=str(tmp_dir))
        return ddpg

    @pytest.fixture(scope="function")
    def disc_ddpg(self, train_config):
        # not used for training, only used for testing apis
        c = train_config
        actor = smw(ActorDiscreet(c.observe_dim, c.action_dim)
                    .to(c.device), c.device, c.device)
        actor_t = smw(ActorDiscreet(c.observe_dim, c.action_dim)
                      .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim, c.action_dim)
                     .to(c.device), c.device, c.device)
        critic_t = smw(Critic(c.observe_dim, c.action_dim)
                       .to(c.device), c.device, c.device)
        ddpg = DDPG(actor, actor_t, critic, critic_t,
                    t.optim.Adam,
                    nn.MSELoss(reduction='sum'),
                    replay_device=c.device,
                    replay_size=c.replay_size)
        return ddpg

    @pytest.fixture(scope="function")
    def lr_ddpg(self, train_config):
        # not used for training, only used for testing apis
        c = train_config
        actor = smw(ActorDiscreet(c.observe_dim, c.action_dim)
                    .to(c.device), c.device, c.device)
        actor_t = smw(ActorDiscreet(c.observe_dim, c.action_dim)
                      .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim, c.action_dim)
                     .to(c.device), c.device, c.device)
        critic_t = smw(Critic(c.observe_dim, c.action_dim)
                       .to(c.device), c.device, c.device)
        lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)],
                                         logger=logger)
        with pytest.raises(TypeError, match="missing .+ positional argument"):
            _ = DDPG(actor, actor_t, critic, critic_t,
                     t.optim.Adam,
                     nn.MSELoss(reduction='sum'),
                     replay_device=c.device,
                     replay_size=c.replay_size,
                     lr_scheduler=LambdaLR)
        ddpg = DDPG(actor, actor_t, critic, critic_t,
                    t.optim.Adam,
                    nn.MSELoss(reduction='sum'),
                    replay_device=c.device,
                    replay_size=c.replay_size,
                    lr_scheduler=LambdaLR,
                    lr_scheduler_args=((lr_func,), (lr_func,)))
        return ddpg

    ########################################################################
    # Test for DDPG contiguous domain acting
    ########################################################################
    def test_contiguous_action(self, train_config, ddpg):
        c = train_config
        state = t.zeros([1, c.observe_dim])
        ddpg.act({"state": state})
        ddpg.act({"state": state}, use_target=True)
        ddpg.act_with_noise({"state": state}, noise_param=(0, 1.0),
                            mode="uniform")
        ddpg.act_with_noise({"state": state}, noise_param=(0, 1.0),
                            mode="normal")
        ddpg.act_with_noise({"state": state}, noise_param=(0, 1.0, -1.0, 1.0),
                            mode="clipped_normal")
        ddpg.act_with_noise({"state": state}, noise_param={"mu": 0, "sigma": 1},
                            mode="ou")
        with pytest.raises(ValueError, match="Unknown noise type"):
            ddpg.act_with_noise({"state": state},
                                noise_param=None,
                                mode="some_unknown_noise")

    ########################################################################
    # Test for DDPG discreet domain acting
    ########################################################################
    def test_discreet_action(self, train_config, disc_ddpg):
        c = train_config
        state = t.zeros([1, c.observe_dim])
        disc_ddpg.act_discreet({"state": state})
        disc_ddpg.act_discreet({"state": state}, use_target=True)
        disc_ddpg.act_discreet_with_noise({"state": state})
        disc_ddpg.act_discreet_with_noise({"state": state}, use_target=True)

    ########################################################################
    # Test for DDPG criticizing
    ########################################################################
    def test_criticize(self, train_config, ddpg):
        c = train_config
        state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, c.action_dim])
        ddpg.criticize({"state": state}, {"action": action})
        ddpg.criticize({"state": state}, {"action": action}, use_target=True)

    ########################################################################
    # Test for DDPG storage
    ########################################################################
    def test_storage(self, train_config, ddpg):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, c.action_dim])
        ddpg.store_transition({
            "state": {"state": old_state.clone()},
            "action": {"action": action.clone()},
            "next_state": {"state": state.clone()},
            "reward": 0,
            "terminal": False
        })
        ddpg.store_episode([{
            "state": {"state": old_state.clone()},
            "action": {"action": action.clone()},
            "next_state": {"state": state.clone()},
            "reward": 0,
            "terminal": False
        }])

    ########################################################################
    # Test for DDPG update
    ########################################################################
    def test_update(self, train_config, ddpg):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, c.action_dim])
        ddpg.store_transition({
            "state": {"state": old_state.clone()},
            "action": {"action": action.clone()},
            "next_state": {"state": state.clone()},
            "reward": 0,
            "terminal": False
        })
        ddpg.update(update_value=True, update_policy=True,
                    update_target=True, concatenate_samples=True)
        ddpg.update(update_value=False, update_policy=False,
                    update_target=False, concatenate_samples=True)

    ########################################################################
    # Test for DDPG save & load
    ########################################################################
    def test_save_load(self, train_config, ddpg, tmpdir):
        save_dir = tmpdir.make_numbered_dir()
        ddpg.save(model_dir=str(save_dir),
                  network_map={
                      "critic_target": "critic_t",
                      "actor_target": "actor_t"
                  },
                  version=1000)
        ddpg.load(model_dir=str(save_dir),
                  network_map={
                      "critic_target": "critic_t",
                      "actor_target": "actor_t"
                  },
                  version=1000)

    ########################################################################
    # Test for DDPG lr_scheduler
    ########################################################################
    def test_lr_scheduler(self, train_config, lr_ddpg):
        lr_ddpg.update_lr_scheduler()

    ########################################################################
    # Test for DDPG full training.
    ########################################################################
    def test_full_train(self, train_config, ddpg):
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
                        action = ddpg.act_with_noise(
                            {"state": old_state.unsqueeze(0)},
                            noise_param=c.noise_param,
                            mode=c.noise_mode
                        )
                    else:
                        action = ddpg.act({"state": old_state.unsqueeze(0)}) \
                            .clamp(-c.action_range, c.action_range)

                    state, reward, terminal, _ = env.step(action.cpu().numpy())
                    state = t.tensor(state, dtype=t.float32, device=c.device) \
                        .flatten()
                    total_reward += float(reward)

                    ddpg.store_transition({
                        "state": {"state": old_state.unsqueeze(0).clone()},
                        "action": {"action": action.clone()},
                        "next_state": {"state": state.unsqueeze(0).clone()},
                        "reward": float(reward),
                        "terminal": terminal or step == c.max_steps
                    })
            # update
            if episode > 100:
                for i in range(step.get()):
                    ddpg.update()

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

        pytest.fail("DDPG Training failed.")
