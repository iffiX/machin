from machin.frame.helpers.servers import model_server_helper
from machin.frame.algorithms import IMPALA
from machin.parallel.distributed import World
from machin.utils.logging import default_logger as logger
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import spawn
from torch.distributions import Categorical
from time import sleep

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
    world = World(world_size=4, rank=rank,
                  name=str(rank), rpc_timeout=20)

    servers = model_server_helper()
    impala_group = world.create_rpc_group("impala", ["0", "1", "2", "3"])

    if rank in (2, 3):
        # learner_group.group is the wrapped torch.distributed.ProcessGroup
        learner_group = world.create_collective_group(ranks=[2, 3])

        # wrap the model with DistributedDataParallel
        # if current process is learner process 2 or 3
        actor = DistributedDataParallel(module=Actor(observe_dim, action_num),
                                        process_group=learner_group.group)
        critic = DistributedDataParallel(module=Critic(observe_dim),
                                         process_group=learner_group.group)
    else:
        actor = Actor(observe_dim, action_num)
        critic = Critic(observe_dim)

    # we may use a smaller batch size to train if we are using
    # DistributedDataParallel

    # note: since the impala framework is storing a whole
    # episode as a single sample, we should wait for a smaller number
    impala = IMPALA(actor, critic,
                    t.optim.Adam,
                    nn.MSELoss(reduction='sum'),
                    impala_group,
                    servers,
                    batch_size=2)

    # synchronize all processes in the group, make sure
    # distributed buffer has been created on all processes in apex_group
    impala_group.barrier()

    # manually control syncing to improve performance
    impala.set_sync(False)
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
            impala.manual_sync()
            tmp_observations = []
            while not terminal and step <= max_steps:
                step += 1
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action, action_log_prob, *_ = \
                        impala.act({"state": old_state})
                    state, reward, terminal, _ = env.step(action.item())
                    state = t.tensor(state, dtype=t.float32) \
                        .view(1, observe_dim)
                    total_reward += reward

                    tmp_observations.append({
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "action_log_prob": action_log_prob.item(),
                        "terminal": terminal or step == max_steps
                    })

            impala.store_episode(tmp_observations)
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
        # note: since the impala framework is storing a whole
        # episode as a single sample, we should wait for a smaller number
        while impala.replay_buffer.all_size() < 5:
            sleep(0.1)
        while True:
            impala.update()


if __name__ == "__main__":
    # spawn 4 sub processes
    # Process 0 and 1 will be workers(samplers)
    # Process 2 and 3 will be learners, using DistributedDataParallel
    spawn(main, nprocs=4)
