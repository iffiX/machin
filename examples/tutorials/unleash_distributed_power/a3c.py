from machin.frame.helpers.servers import grad_server_helper
from machin.frame.algorithms import A3C
from machin.parallel.distributed import World
from machin.utils.logging import default_logger as logger
from torch.multiprocessing import spawn
from torch.distributions import Categorical
import gym
import torch as t
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_num):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = (action
               if action is not None
               else dist.sample())
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


def main(rank):
    env = gym.make("CartPole-v0")
    observe_dim = 4
    action_num = 2
    max_episodes = 2000
    max_steps = 200
    solved_reward = 190
    solved_repeat = 5

    # initlize distributed world first
    _world = World(world_size=3, rank=rank,
                   name=str(rank), rpc_timeout=20)

    actor = Actor(observe_dim, action_num)
    critic = Critic(observe_dim)

    # in all test scenarios, all processes will be used as reducers
    servers = grad_server_helper(
        [lambda: Actor(observe_dim, action_num),
         lambda: Critic(observe_dim)],
        learning_rate=5e-3
    )
    a3c = A3C(actor, critic,
              nn.MSELoss(reduction='sum'),
              servers)

    # manually control syncing to improve performance
    a3c.set_sync(False)

    # begin training
    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0

        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

        # manually pull the newest parameters
        a3c.manual_sync()
        tmp_observations = []
        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = a3c.act({"state": old_state})[0]
                state, reward, terminal, _ = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward

                tmp_observations.append({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps
                })

        # update
        a3c.store_episode(tmp_observations)
        a3c.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)
        logger.info("Process {} Episode {} total reward={:.2f}"
                    .format(rank, episode, smoothed_total_reward))

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                # will cause torch RPC to complain
                # since other processes may have not finished yet.
                # just for demonstration.
                exit(0)
        else:
            reward_fulfilled = 0


if __name__ == "__main__":
    # spawn 3 sub processes
    spawn(main, nprocs=3)
