from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.sac import SAC
from machin.utils.learning_rate import gen_learning_rate_func
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import softplus
from torch.distributions import Normal

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
        self.mu_head = nn.Linear(16, action_dim)
        self.sigma_head = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        mu = t.tanh(self.mu_head(a)) * self.action_range
        sigma = softplus(self.sigma_head(a))
        dist = Normal(mu, sigma)
        act = (action
               if action is not None
               else dist.rsample())
        act_entropy = dist.entropy()
        # do not clamp actions here, because
        # we action probability might be extremely small,
        # and network will "nan".
        act_log_prob = dist.log_prob(act)

        # do not clamp actions here, because
        # actions will be stored in replay buffer
        # and new evaluated log probability in update
        # might also be extremely small, and network will "nan".

        # clamp actions only before sending your actions into
        # the environment.
        return act, act_log_prob, act_entropy


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


class TestSAC(object):
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
        c.replay_size = 100000
        c.solved_reward = -150
        c.solved_repeat = 5
        c.device = "cpu"
        return c

    @pytest.fixture(scope="function")
    def sac(self, train_config):
        c = train_config
        actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim, c.action_dim)
                     .to(c.device), c.device, c.device)
        critic_t = smw(Critic(c.observe_dim, c.action_dim)
                       .to(c.device), c.device, c.device)
        critic2 = smw(Critic(c.observe_dim, c.action_dim)
                      .to(c.device), c.device, c.device)
        critic2_t = smw(Critic(c.observe_dim, c.action_dim)
                        .to(c.device), c.device, c.device)
        sac = SAC(actor, critic, critic_t, critic2, critic2_t,
                  t.optim.Adam,
                  nn.MSELoss(reduction='sum'),
                  replay_device=c.device,
                  replay_size=c.replay_size)
        return sac

    @pytest.fixture(scope="function")
    def sac_vis(self, train_config, tmpdir):
        # not used for training, only used for testing apis
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim, c.action_dim)
                     .to(c.device), c.device, c.device)
        critic_t = smw(Critic(c.observe_dim, c.action_dim)
                       .to(c.device), c.device, c.device)
        critic2 = smw(Critic(c.observe_dim, c.action_dim)
                      .to(c.device), c.device, c.device)
        critic2_t = smw(Critic(c.observe_dim, c.action_dim)
                        .to(c.device), c.device, c.device)
        sac = SAC(actor, critic, critic_t, critic2, critic2_t,
                  t.optim.Adam,
                  nn.MSELoss(reduction='sum'),
                  replay_device=c.device,
                  replay_size=c.replay_size,
                  visualize=True,
                  visualize_dir=str(tmp_dir))
        return sac

    @pytest.fixture(scope="function")
    def lr_sac(self, train_config):
        # not used for training, only used for testing apis
        c = train_config
        actor = smw(Actor(c.observe_dim, c.action_dim, c.action_range)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim, c.action_dim)
                     .to(c.device), c.device, c.device)
        critic_t = smw(Critic(c.observe_dim, c.action_dim)
                       .to(c.device), c.device, c.device)
        critic2 = smw(Critic(c.observe_dim, c.action_dim)
                      .to(c.device), c.device, c.device)
        critic2_t = smw(Critic(c.observe_dim, c.action_dim)
                        .to(c.device), c.device, c.device)
        lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)],
                                         logger=logger)
        with pytest.raises(TypeError, match="missing .+ positional argument"):
            _ = SAC(actor, critic, critic_t, critic2, critic2_t,
                    t.optim.Adam,
                    nn.MSELoss(reduction='sum'),
                    replay_device=c.device,
                    replay_size=c.replay_size,
                    lr_scheduler=LambdaLR)
        sac = SAC(actor, critic, critic_t, critic2, critic2_t,
                  t.optim.Adam,
                  nn.MSELoss(reduction='sum'),
                  replay_device=c.device,
                  replay_size=c.replay_size,
                  lr_scheduler=LambdaLR,
                  lr_scheduler_args=((lr_func,), (lr_func,), (lr_func,)))
        return sac

    ########################################################################
    # Test for SAC acting
    ########################################################################
    def test_action(self, train_config, sac):
        c = train_config
        state = t.zeros([1, c.observe_dim])
        sac.act({"state": state})

    ########################################################################
    # Test for SAC criticizing
    ########################################################################
    def test_criticize(self, train_config, sac):
        c = train_config
        state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, c.action_dim])
        sac.criticize({"state": state}, {"action": action})
        sac.criticize({"state": state}, {"action": action}, use_target=True)
        sac.criticize2({"state": state}, {"action": action})
        sac.criticize2({"state": state}, {"action": action}, use_target=True)

    ########################################################################
    # Test for SAC storage
    ########################################################################
    def test_store(self, train_config, sac):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, c.action_dim])
        sac.store_transition({
            "state": {"state": old_state.clone()},
            "action": {"action": action.clone()},
            "next_state": {"state": state.clone()},
            "reward": 0,
            "terminal": False
        })
        sac.store_episode([{
            "state": {"state": old_state.clone()},
            "action": {"action": action.clone()},
            "next_state": {"state": state.clone()},
            "reward": 0,
            "terminal": False
        }])

    ########################################################################
    # Test for SAC update
    ########################################################################
    def test_update(self, train_config, sac_vis):
        c = train_config
        old_state = state = t.zeros([1, c.observe_dim])
        action = t.zeros([1, c.action_dim])
        sac_vis.store_transition({
            "state": {"state": old_state.clone()},
            "action": {"action": action.clone()},
            "next_state": {"state": state.clone()},
            "reward": 0,
            "terminal": False
        })
        # heuristic entropy
        sac_vis.target_entropy = -c.action_dim
        sac_vis.update(update_value=True, update_policy=True,
                       update_target=True, update_entropy_alpha=True,
                       concatenate_samples=True)
        sac_vis.update(update_value=False, update_policy=False,
                       update_target=False, update_entropy_alpha=False,
                       concatenate_samples=True)

    ########################################################################
    # Test for SAC save & load
    ########################################################################
    def test_save_load(self, train_config, sac, tmpdir):
        save_dir = tmpdir.make_numbered_dir()
        sac.save(model_dir=str(save_dir),
                 network_map={
                     "critic_target": "critic_t",
                     "critic2_target": "critic2_t",
                     "actor": "actor"
                 },
                 version=1000)
        sac.load(model_dir=str(save_dir),
                 network_map={
                     "critic_target": "critic_t",
                     "critic2_target": "critic2_t",
                     "actor": "actor"
                 },
                 version=1000)

    ########################################################################
    # Test for SAC lr_scheduler
    ########################################################################
    def test_lr_scheduler(self, train_config, lr_sac):
        lr_sac.update_lr_scheduler()

    ########################################################################
    # Test for SAC full training.
    ########################################################################
    def test_full_train(self, train_config, sac):
        c = train_config
        sac.target_entropy = c.action_dim

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
                    action = sac.act({"state": old_state.unsqueeze(0)})[0]

                    state, reward, terminal, _ = env.step(
                        action.clamp(-c.action_range, c.action_range)
                              .cpu().numpy()
                    )
                    state = t.tensor(state, dtype=t.float32, device=c.device) \
                        .flatten()
                    total_reward += float(reward)

                    sac.store_transition({
                        "state": {"state": old_state.unsqueeze(0).clone()},
                        "action": {"action": action.clone()},
                        "next_state": {"state": state.unsqueeze(0).clone()},
                        "reward": float(reward),
                        "terminal": terminal or step == c.max_steps
                    })
            # update
            if episode > 100:
                for i in range(step.get()):
                    sac.update()

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

        pytest.fail("SAC Training failed.")
