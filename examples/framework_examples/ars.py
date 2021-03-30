from machin.model.nets.base import dynamic_module_wrapper as dmw
from machin.frame.helpers.servers import model_server_helper
from machin.frame.algorithms import ARS
from machin.parallel.distributed import World
from machin.utils.logging import default_logger as logger
from torch.multiprocessing import spawn
import gym
import torch as t
import torch.nn as nn


class ActorDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, action_dim, bias=False)

    def forward(self, state):
        a = t.argmax(self.fc(state), dim=1).item()
        return a


def main(rank):
    env = gym.make("CartPole-v0")
    observe_dim = 4
    action_num = 2
    max_episodes = 2000
    max_steps = 200
    solved_reward = 190
    solved_repeat = 5

    # initlize distributed world first
    world = World(world_size=3, rank=rank, name=str(rank), rpc_timeout=20)

    actor = dmw(ActorDiscrete(observe_dim, action_num))
    servers = model_server_helper(model_num=1)
    ars_group = world.create_rpc_group("ars", ["0", "1", "2"])
    ars = ARS(
        actor,
        t.optim.SGD,
        ars_group,
        servers,
        noise_std_dev=0.1,
        learning_rate=0.1,
        noise_size=1000000,
        rollout_num=6,
        used_rollout_num=6,
        normalize_state=True,
    )

    # begin training
    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        all_reward = 0
        for at in ars.get_actor_types():
            total_reward = 0
            terminal = False
            step = 0

            # batch size = 1
            state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
            while not terminal and step <= max_steps:
                step += 1
                with t.no_grad():
                    # agent model inference
                    action = ars.act({"state": state}, at)
                    state, reward, terminal, __ = env.step(action)
                    state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                    total_reward += reward

            ars.store_reward(total_reward, at)
            all_reward += total_reward

        # update
        ars.update()

        # show reward
        smoothed_total_reward = (
            smoothed_total_reward * 0.9 + all_reward / len(ars.get_actor_types()) * 0.1
        )
        logger.info(
            f"Process {rank} Episode {episode} total reward={smoothed_total_reward:.2f}"
        )

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
