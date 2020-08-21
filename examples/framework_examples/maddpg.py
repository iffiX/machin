from machin.frame.algorithms import MADDPG
from machin.utils.logging import default_logger as logger
from copy import deepcopy
import torch as t
import torch.nn as nn


# Important note:
# In order to successfully run the environment, please git clone the project
# then run:
#    pip install -e ./test_lib/multiagent-particle-envs/
# in project root directory


def create_env(env_name):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(env_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, info_callback=None,
                        shared_viewer=False)
    return env


# configurations
env = create_env("simple_spread")
env.discrete_action_input = True
observe_dim = env.observation_space[0].shape[0]
action_num = env.action_space[0].n
max_episodes = 1000
max_steps = 200
# number of agents in env, fixed, do not change
agent_num = 3
solved_reward = -15
solved_repeat = 5


# model definition
class ActorDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorDiscrete, self).__init__()

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


if __name__ == "__main__":
    actor = ActorDiscrete(observe_dim, action_num)
    critic = Critic(observe_dim * agent_num,
                    action_num * agent_num)

    maddpg = MADDPG([deepcopy(actor) for _ in range(agent_num)],
                    [deepcopy(actor) for _ in range(agent_num)],
                    [deepcopy(critic) for _ in range(agent_num)],
                    [deepcopy(critic) for _ in range(agent_num)],
                    [list(range(agent_num))] * agent_num,
                    t.optim.Adam,
                    nn.MSELoss(reduction='sum'))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        states = [t.tensor(st, dtype=t.float32).view(1, observe_dim)
                  for st in env.reset()]

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_states = states
                # agent model inference
                results = maddpg.act_discrete_with_noise(
                    [{"state": st} for st in states]
                )
                actions = [int(r[0]) for r in results]
                action_probs = [r[1] for r in results]

                states, rewards, terminals, _ = env.step(actions)
                states = [t.tensor(st, dtype=t.float32).view(1, observe_dim)
                          for st in states]
                total_reward += float(sum(rewards)) / agent_num

                maddpg.store_transitions([{
                    "state": {"state": ost},
                    "action": {"action": act},
                    "next_state": {"state": st},
                    "reward": float(rew),
                    "terminal": term or step == max_steps
                } for ost, act, st, rew, term in zip(
                    old_states, action_probs, states, rewards, terminals
                )])

        # total reward is divided by steps here, since:
        # "Agents are rewarded based on minimum agent distance
        #  to each landmark, penalized for collisions"
        total_reward /= step

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                maddpg.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)
        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))

        if smoothed_total_reward > solved_reward and episode > 100:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0
