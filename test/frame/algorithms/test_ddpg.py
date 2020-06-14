from machin.model.nets.base import static_module_wrapper
from machin.frame.algorithms.ddpg import DDPG
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter, Timer
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window

import pytest
import torch as t
import torch.nn as nn
import gym


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

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
    ########################################################################
    # Test for DDPG full training.
    ########################################################################
    # configs and definitions
    @pytest.fixture(scope="class")
    def train_config(self, pytestconfig):
        disable_view_window()
        c = Config()
        c.env_name = "CartPole-v1"
        c.env = gym.make(c.env_name)
        c.observe_dim = 4
        c.action_dim = 2

        c.max_episodes = 5000
        c.max_steps = 200
        c.replay_size = 10000

        c.device = "cpu"
        return c

    def test_full_train(self, train_config):
        c = train_config
        actor = static_module_wrapper(Actor(c.observe_dim, c.action_dim)
                                      .to(c.device), c.device, c.device)
        actor_t = static_module_wrapper(Actor(c.observe_dim, c.action_dim)
                                        .to(c.device), c.device, c.device)
        critic = static_module_wrapper(Critic(c.observe_dim, c.action_dim)
                                       .to(c.device), c.device, c.device)
        critic_t = static_module_wrapper(Critic(c.observe_dim, c.action_dim)
                                         .to(c.device), c.device, c.device)
        logger.info("[DDPG full train] Networks created")

        ddpg = DDPG(actor, actor_t, critic, critic_t,
                    t.optim.Adam,
                    nn.MSELoss(reduction='sum'),
                    replay_device=c.device,
                    replay_size=c.replay_size)

        logger.info("[DDPG full train] DDPG framework initialized")

        # begin training
        episode, step = Counter(), Counter()
        episode_timer = Timer()
        terminal = False

        env = c.env
        while episode < c.max_episodes:
            episode.count()
            episode_timer.begin()

            # batch size = 1
            total_reward = 0
            reward = 0
            state = t.tensor(env.reset(), dtype=t.float32, device=c.device)

            tmp_observe = []
            while not terminal and step <= c.max_steps:
                step.count()
                with t.no_grad():
                    old_state = state

                    # agent model inference
                    action = ddpg.act_with_noise(
                        {"state": state.unsqueeze(0)},
                        noise_param=(0.0, 0.3, -0.5, 0.5),
                        mode="clipped_normal"
                    )

                    state, reward, terminal, _ = \
                        env.step(t.argmax(t.softmax(action, dim=1), dim=1)
                                 .item())
                    state = t.tensor(state, dtype=t.float32, device=c.device)
                    total_reward += reward

                    ddpg.store_transition(
                        {"state": {"state": old_state.unsqueeze(0).clone()},
                         "action": {"action": action.clone()},
                         "next_state": {"state": state.unsqueeze(0).clone()},
                         "reward": float(reward),
                         "terminal": terminal or step == c.max_steps}
                    )
            # update
            if episode > 100:
                ddpg.update()

            step_time = episode_timer.end() / step.get()
            episode_time = episode_timer.end()
            step.reset()
            terminal = False
            logger.info("Episode {}: "
                        "step time={:.3f}s, "
                        "total time={:.2f}s, "
                        "total reward={:.2f}"
                        .format(
                            episode, step_time, episode_time, total_reward
                        ))

