from machin.frame.helpers.servers import model_server_helper
from machin.frame.algorithms import DDPGApex
from machin.parallel.distributed import World
from machin.utils.logging import default_logger as logger
from torch.multiprocessing import spawn
from time import sleep

import gym
import torch as t
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        """
        Initialize the gradient.

        Args:
            self: (todo): write your description
            state_dim: (int): write your description
            action_dim: (str): write your description
            action_range: (str): write your description
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        """
        Perform forward.

        Args:
            self: (todo): write your description
            state: (todo): write your description
        """
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the internal state.

        Args:
            self: (todo): write your description
            state_dim: (int): write your description
            action_dim: (str): write your description
        """
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        """
        Perform forward forward computation.

        Args:
            self: (todo): write your description
            state: (todo): write your description
            action: (str): write your description
        """
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


def main(rank):
    """
    Main function.

    Args:
        rank: (int): write your description
    """
    env = gym.make("Pendulum-v0")
    observe_dim = 3
    action_dim = 1
    action_range = 2
    max_episodes = 2000
    max_steps = 200
    noise_param = (0, 0.2)
    noise_mode = "normal"
    solved_reward = -150
    solved_repeat = 5

    # initlize distributed world first
    world = World(world_size=4, rank=rank,
                  name=str(rank), rpc_timeout=20)

    servers = model_server_helper(model_num=2)
    apex_group = world.create_rpc_group("apex", ["0", "1", "2", "3"])

    actor = Actor(observe_dim, action_dim, action_range)
    actor_t = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)

    ddpg_apex = DDPGApex(actor, actor_t, critic, critic_t,
                         t.optim.Adam,
                         nn.MSELoss(reduction='sum'),
                         apex_group,
                         servers)

    # synchronize all processes in the group, make sure
    # distributed buffer has been created on all processes in apex_group
    apex_group.barrier()

    # manually control syncing to improve performance
    ddpg_apex.set_sync(False)
    if rank in (0, 1):
        # Process 0 and 1 are workers(samplers)
        # begin training
        episode, step, reward_fulfilled = 0, 0, 0
        smoothed_total_reward = 0

        while episode < max_episodes:
            # sleep to wait for learners keep up
            sleep(0.1)
            episode += 1
            total_reward = 0
            terminal = False
            step = 0

            state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

            # manually pull the newest parameters
            ddpg_apex.manual_sync()
            while not terminal and step <= max_steps:
                step += 1
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action = ddpg_apex.act_with_noise(
                            {"state": old_state},
                            noise_param=noise_param,
                            mode=noise_mode
                        )
                    state, reward, terminal, _ = env.step(action.numpy())
                    state = t.tensor(state, dtype=t.float32)\
                        .view(1, observe_dim)
                    total_reward += reward[0]

                    ddpg_apex.store_transition({
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward[0],
                        "terminal": terminal or step == max_steps
                    })

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

    elif rank in (2, 3):
        # wait for enough samples
        while ddpg_apex.replay_buffer.all_size() < 500:
            sleep(0.1)
        while True:
            ddpg_apex.update()


if __name__ == "__main__":
    # spawn 4 sub processes
    # Process 0 and 1 will be workers(samplers)
    # Process 2 and 3 will be learners, using DistributedDataParallel
    spawn(main, nprocs=4)
