from machin.model.nets.base import static_module_wrapper as smw
from machin.frame.algorithms.maddpg import MADDPG
#from machin.frame.algorithms.maddpg_jit import MADDPG
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

from .utils import Smooth

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


class ActorDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorDiscrete, self).__init__()

        self.fc1 = nn.Linear(state_dim, 160)
        self.fc2 = nn.Linear(160, 160)
        self.fc3 = nn.Linear(160, action_dim)

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


class TestMADDPG(object):
    # configs and definitions
    @pytest.fixture(scope="class")
    def train_config(self, pytestconfig):
        disable_view_window()
        c = Config()
        # the predator-prey environment provided in
        # https://github.com/openai/multiagent-particle-envs
        c.env_name = "simple_tag"
        c.env = create_env(c.env_name)
        c.env.discrete_action_input = True
        c.pred_num = 3
        c.prey_num = 1
        # first three agents are predators,
        # the last one is prey
        c.prey_observe_dim = c.env.observation_space[3].shape[0]
        c.pred_observe_dim = c.env.observation_space[0].shape[0]
        c.prey_action_num = c.env.action_space[3].n
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
        c.solved_reward = [-20, 10]
        c.solved_repeat = 5
        c.device = "cpu"
        return c

    @pytest.fixture(scope="function")
    def maddpg(self, train_config):
        c = train_config
        # for simplicity, prey will be trained with predators,
        # Predator can get the observation of prey, same for prey.
        pred_actor = smw(ActorDiscrete(c.pred_observe_dim,
                                       c.pred_action_num)
                         .to(c.device), c.device, c.device)
        prey_actor = smw(ActorDiscrete(c.prey_observe_dim,
                                       c.prey_action_num)
                         .to(c.device), c.device, c.device)
        pred_critic = smw(Critic(c.pred_observe_dim * 3,
                                 c.pred_action_num * 3)
                          .to(c.device), c.device, c.device)
        prey_critic = smw(Critic(c.prey_observe_dim, c.prey_action_num)
                          .to(c.device), c.device, c.device)
        # first three agents are preds, the last agent is prey
        # Thread pool is more efficient because the models are small.
        maddpg = MADDPG([deepcopy(pred_actor) for _ in range(3)] +
                        [deepcopy(prey_actor)],
                        [deepcopy(pred_actor) for _ in range(3)] +
                        [deepcopy(prey_actor)],
                        [deepcopy(pred_critic) for _ in range(3)] +
                        [deepcopy(prey_critic)],
                        [deepcopy(pred_critic) for _ in range(3)] +
                        [deepcopy(prey_critic)],
                        [[0, 1, 2], [0, 1, 2], [0, 1, 2], [3]],
                        t.optim.Adam,
                        nn.MSELoss(reduction='sum'),
                        replay_device=c.device,
                        replay_size=c.replay_size,
                        pool_type="thread")
        return maddpg

    @pytest.fixture(scope="function")
    def maddpg_disc(self, train_config):
        c = train_config
        actor = smw(ActorDiscrete(c.test_observe_dim, c.test_action_dim)
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
    # Test for MADDPG discrete domain acting
    ########################################################################
    def test_discrete_act(self, train_config, maddpg_disc):
        c = train_config
        states = ([{"state": t.zeros([1, c.test_observe_dim])}]
                  * c.test_agent_num)
        maddpg_disc.act_discrete(states)
        maddpg_disc.act_discrete(states, use_target=True)
        maddpg_disc.act_discrete_with_noise(states)
        maddpg_disc.act_discrete_with_noise(states, use_target=True)

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
    def test_full_train(self, train_config, maddpg):
        c = train_config

        # begin training
        episode, step = Counter(), Counter()

        # first for prey, second for pred
        smoothers = [Smooth(), Smooth()]
        fulfilled = [Counter(), Counter()]
        solved = [False, False]
        terminal = False

        env = c.env
        while episode < c.max_episodes:
            episode.count()

            # batch size = 1
            pred_total_reward = 0
            prey_total_reward = 0
            states = [t.tensor(st, dtype=t.float32, device=c.device)
                      for st in env.reset()]

            while not terminal and step <= c.max_steps:
                step.count()
                with t.no_grad():
                    old_states = states

                    # agent model inference
                    actions, action_probs = maddpg.act_discrete_with_noise(
                        [{"state": st.unsqueeze(0)} for st in states]
                    )
                    actions = [int(act) for act in actions]
                    states, rewards, terminals, _ = env.step(actions)
                    states = [t.tensor(st, dtype=t.float32, device=c.device)
                              for st in states]
                    pred_total_reward += float(sum(rewards[:3])) / 3
                    prey_total_reward += float(rewards[3])

                    maddpg.store_transitions([{
                        "state": {"state": ost.unsqueeze(0).clone()},
                        "action": {"action": act.clone()},
                        "next_state": {"state": st.unsqueeze(0).clone()},
                        "reward": float(rew),
                        "terminal": term or step == c.max_steps
                    } for ost, act, st, rew, term in zip(
                        old_states, action_probs, states, rewards, terminals
                    )])

            # update
            if episode > 5:
                for i in range(step.get()):
                    maddpg.update()

            smoothers[0].update(prey_total_reward)
            smoothers[1].update(pred_total_reward)
            step.reset()
            terminal = False

            logger.info("Episode {} prey total reward={:.2f}, "
                        "pred_total_reward={:.2f}"
                        .format(episode,
                                smoothers[0].value,
                                smoothers[1].value))

            for smoother, idx, name in zip(smoothers, [0, 1],
                                           ["prey", "pred"]):
                if smoother.value > c.solved_reward[idx]:
                    fulfilled[idx].count()
                    if fulfilled[idx] >= c.solved_repeat and not solved[idx]:
                        solved[idx] = True
                        logger.info("Environment {} solved!".format(name))
                else:
                    fulfilled[idx].reset()

            if all(solved):
                return

        pytest.fail("MADDPG Training failed.")
