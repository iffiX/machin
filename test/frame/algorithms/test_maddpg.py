from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.maddpg import MADDPG
from machin.utils.learning_rate import gen_learning_rate_func
from machin.utils.logging import default_logger as logger
from machin.utils.helper_classes import Counter
from machin.utils.conf import Config
from machin.env.utils.openai_gym import disable_view_window
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy

import pytest
import torch as t
import torch.nn as nn
import gym

from test.util_create_ma_env import create_env


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
        # This critic implementation is shared by the prey(DDPG) and
        # predators(MADDPG)
        # Note: For MADDPG
        #       state_dim is the dimension of all states from all agents.
        #       action_dim is the dimension of all actions from all agents.
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
        # the predator-prey environment provided in
        # https://github.com/openai/multiagent-particle-envs
        c.env_name = "simple_tag"
        c.env = create_env(c.env_name)
        c.pred_num = 3
        c.prey_num = 1
        # first three agents are predators,
        # the last one is prey
        # these observation dims include all states of all agents
        c.prey_observe_dim = c.env.observation_space[3].shape[0]
        c.prey_action_num = c.env.action_space[3].n
        c.pred_observe_dim = c.env.observation_space[0].shape[0]
        c.pred_action_num = c.env.action_space[0].n
        # for contiguous tests
        c.test_action_dim = 5
        c.test_action_range = 1
        c.test_observe_dim = 5
        c.test_agent_num = 3
        c.max_episodes = 1000
        c.max_steps = 200
        c.replay_size = 100000
        # from https://github.com/wsjeon/maddpg-rllib/tree/master/plots
        c.pred_solved_reward = 10
        c.prey_solved_reward = -20
        c.solved_repeat = 5
        c.device = "cpu"
        return c

    @pytest.fixture(scope="function")
    def maddpg_full(self, train_config):
        c = train_config
        actor = smw(ActorDiscreet(c.observe_dim, c.action_dim)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.observe_dim, c.action_dim)
                     .to(c.device), c.device, c.device)
        maddpg = MADDPG([deepcopy(actor) for _ in range(c.test_agent_num)],
                        [deepcopy(actor) for _ in range(c.test_agent_num)],
                        [deepcopy(critic) for _ in range(c.test_agent_num)],
                        [deepcopy(critic) for _ in range(c.test_agent_num)],
                        t.optim.Adam,
                        nn.MSELoss(reduction='sum'),
                        replay_device=c.device,
                        replay_size=c.replay_size)
        return maddpg

    @pytest.fixture(scope="function")
    def maddpg_disc(self, train_config):
        c = train_config
        actor = smw(ActorDiscreet(c.test_observe_dim, c.test_action_dim)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.test_observe_dim * c.test_agent_num,
                            c.test_action_dim * c.test_agent_num)
                     .to(c.device), c.device, c.device)

        maddpg = MADDPG([deepcopy(actor) for _ in range(c.test_agent_num)],
                        [deepcopy(actor) for _ in range(c.test_agent_num)],
                        [deepcopy(critic) for _ in range(c.test_agent_num)],
                        [deepcopy(critic) for _ in range(c.test_agent_num)],
                        t.optim.Adam,
                        nn.MSELoss(reduction='sum'),
                        replay_device=c.device,
                        replay_size=c.replay_size)
        return maddpg

    @pytest.fixture(scope="function")
    def maddpg_cont(self, train_config):
        c = train_config
        actor = smw(Actor(c.test_observe_dim, c.test_action_dim,
                          c.test_action_range)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.test_observe_dim * c.test_agent_num,
                            c.test_action_dim * c.test_agent_num)
                     .to(c.device), c.device, c.device)

        maddpg = MADDPG([deepcopy(actor) for _ in range(c.test_agent_num)],
                        [deepcopy(actor) for _ in range(c.test_agent_num)],
                        [deepcopy(critic) for _ in range(c.test_agent_num)],
                        [deepcopy(critic) for _ in range(c.test_agent_num)],
                        t.optim.Adam,
                        nn.MSELoss(reduction='sum'),
                        replay_device=c.device,
                        replay_size=c.replay_size)
        return maddpg

    @pytest.fixture(scope="function")
    def maddpg_vis(self, train_config, tmpdir):
        c = train_config
        tmp_dir = tmpdir.make_numbered_dir()
        actor = smw(Actor(c.test_observe_dim, c.test_action_dim,
                          c.test_action_range)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.test_observe_dim * c.test_agent_num,
                            c.test_action_dim * c.test_agent_num)
                     .to(c.device), c.device, c.device)

        maddpg = MADDPG([deepcopy(actor) for _ in range(c.test_agent_num)],
                        [deepcopy(actor) for _ in range(c.test_agent_num)],
                        [deepcopy(critic) for _ in range(c.test_agent_num)],
                        [deepcopy(critic) for _ in range(c.test_agent_num)],
                        t.optim.Adam,
                        nn.MSELoss(reduction='sum'),
                        replay_device=c.device,
                        replay_size=c.replay_size,
                        visualize=True,
                        visualize_dir=str(tmp_dir))
        return maddpg

    @pytest.fixture(scope="function")
    def maddpg_lr(self, train_config):
        c = train_config
        actor = smw(Actor(c.test_observe_dim, c.test_action_dim,
                          c.test_action_range)
                    .to(c.device), c.device, c.device)
        critic = smw(Critic(c.test_observe_dim * c.test_agent_num,
                            c.test_action_dim * c.test_agent_num)
                     .to(c.device), c.device, c.device)
        lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)],
                                         logger=logger)
        with pytest.raises(TypeError, match="missing .+ positional argument"):
            _ = MADDPG([deepcopy(actor) for _ in range(c.test_agent_num)],
                       [deepcopy(actor) for _ in range(c.test_agent_num)],
                       [deepcopy(critic) for _ in range(c.test_agent_num)],
                       [deepcopy(critic) for _ in range(c.test_agent_num)],
                       t.optim.Adam,
                       nn.MSELoss(reduction='sum'),
                       replay_device=c.device,
                       replay_size=c.replay_size,
                       lr_scheduler=LambdaLR)
        maddpg = MADDPG([deepcopy(actor) for _ in range(c.test_agent_num)],
                        [deepcopy(actor) for _ in range(c.test_agent_num)],
                        [deepcopy(critic) for _ in range(c.test_agent_num)],
                        [deepcopy(critic) for _ in range(c.test_agent_num)],
                        t.optim.Adam,
                        nn.MSELoss(reduction='sum'),
                        replay_device=c.device,
                        replay_size=c.replay_size,
                        lr_scheduler=LambdaLR,
                        lr_scheduler_args=((lr_func,), (lr_func,)))
        return maddpg

    ########################################################################
    # Test for MADDPG contiguous domain acting
    ########################################################################
    def test_contiguous_act(self, train_config, maddpg_cont):
        c = train_config
        states = ([{"state": t.zeros([1, c.test_observe_dim])}]
                  * c.test_agent_num)
        maddpg_cont.act(states)
        maddpg_cont.act(states, use_target=True)
        maddpg_cont.act_with_noise(states, noise_param=(0, 1.0),
                                   mode="uniform")
        maddpg_cont.act_with_noise(states, noise_param=(0, 1.0),
                                   mode="normal")
        maddpg_cont.act_with_noise(states, noise_param=(0, 1.0, -1.0, 1.0),
                                   mode="clipped_normal")
        maddpg_cont.act_with_noise(states, noise_param={"mu": 0, "sigma": 1},
                                   mode="ou")
        with pytest.raises(ValueError, match="Unknown noise type"):
            maddpg_cont.act_with_noise(states, noise_param=None,
                                       mode="some_unknown_noise")

    ########################################################################
    # Test for MADDPG discreet domain acting
    ########################################################################
    def test_discreet_act(self, train_config, maddpg_disc):
        c = train_config
        states = ([{"state": t.zeros([1, c.test_observe_dim])}]
                  * c.test_agent_num)
        maddpg_disc.act_discreet(states)
        maddpg_disc.act_discreet(states, use_target=True)
        maddpg_disc.act_discreet_with_noise(states)
        maddpg_disc.act_discreet_with_noise(states, use_target=True)

    ########################################################################
    # Test for MADDPG criticizing
    ########################################################################
    def test_criticize(self, train_config, maddpg_cont):
        c = train_config
        states = ([{"state": t.zeros([1, c.test_observe_dim])}]
                  * c.test_agent_num)
        actions = ([{"action": t.zeros([1, c.test_action_dim])}]
                   * c.test_agent_num)
        maddpg_cont.criticize(states, actions, 0)
        maddpg_cont.criticize(states, actions, 1, use_target=True)

    ########################################################################
    # Test for MADDPG storage
    ########################################################################
    def test_store(self, train_config, maddpg_cont):
        c = train_config
        old_state = state = t.zeros([1, c.test_observe_dim])
        action = t.zeros([1, c.test_action_dim])
        maddpg_cont.store_transitions([{
            "state": {"state": old_state.clone()},
            "action": {"action": action.clone()},
            "next_state": {"state": state.clone()},
            "reward": 0,
            "terminal": False
        }] * c.test_agent_num)
        maddpg_cont.store_episodes([[{
            "state": {"state": old_state.clone()},
            "action": {"action": action.clone()},
            "next_state": {"state": state.clone()},
            "reward": 0,
            "terminal": False
        }]] * c.test_agent_num)

    ########################################################################
    # Test for MADDPG update
    ########################################################################
    def test_update(self, train_config, maddpg_cont):
        c = train_config
        old_state = state = t.zeros([1, c.test_observe_dim])
        action = t.zeros([1, c.test_action_dim])
        maddpg_cont.store_episodes([[{
            "state": {"state": old_state.clone()},
            "action": {"action": action.clone()},
            "next_state": {"state": state.clone()},
            "reward": 0,
            "terminal": False
        }]] * c.test_agent_num)
        maddpg_cont.update(update_value=True, update_policy=True,
                           update_target=True, concatenate_samples=True)

    def test_vis_update(self, train_config, maddpg_vis):
        c = train_config
        old_state = state = t.zeros([1, c.test_observe_dim])
        action = t.zeros([1, c.test_action_dim])
        maddpg_vis.store_episodes([[{
            "state": {"state": old_state.clone()},
            "action": {"action": action.clone()},
            "next_state": {"state": state.clone()},
            "reward": 0,
            "terminal": False
        }]] * c.test_agent_num)
        maddpg_vis.update(update_value=True, update_policy=True,
                          update_target=True, concatenate_samples=True)

    ########################################################################
    # Test for MADDPG save & load
    ########################################################################
    def test_save_load(self, train_config, maddpg_cont, tmpdir):
        save_dir = tmpdir.make_numbered_dir()
        maddpg_cont.save(model_dir=str(save_dir),
                         network_map={
                             "critic_target": "critic_t",
                             "actor_target": "actor_t"
                         },
                         version=1000)
        maddpg_cont.load(model_dir=str(save_dir),
                         network_map={
                             "critic_target": "critic_t",
                             "actor_target": "actor_t"
                         },
                         version=1000)

    ########################################################################
    # Test for MADDPG lr_scheduler
    ########################################################################
    def test_lr_scheduler(self, train_config, maddpg_lr):
        maddpg_lr.update_lr_scheduler()

    ########################################################################
    # Test for MADDPG full training.
    ########################################################################
    def tes_full_train(self, train_config, ddpg):
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
