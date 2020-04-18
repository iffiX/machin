import time
import torch as t
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR

from models.frameworks.ddpg import DDPG
from models.tcdn.actor import SwarmActor, WrappedActorNet
from models.tcdn.critic import SwarmCritic, WrappedCriticNet
from models.tcdn.negotiatior import SwarmNegotiator
from models.tcdn.agent import SwarmAgent

from utils.logging import default_logger as logger
from utils.image import create_gif
from utils.tensor_board import global_board
from utils.helper_classes import Counter
from utils.prep import prep_dir_default
from utils.args import get_args

from env.walker.multi_walker import BipedalMultiWalker

# configs
restart = True
# max_batch = 8
max_epochs = 20
max_episodes = 800
max_steps = 1000
theta = 1
replay_size = 2000000
history_depth = 10
agent_num = 5
neighbors = [-1, 1]
noise_range = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
mean_anneal = 0.3
theta_anneal = 0.1
device = t.device("cuda:0")
root_dir = "/data/AI/tmp/multi_agent/walker/tcdn/"
model_dir = root_dir + "model/"
log_dir = root_dir + "log/"
save_map = {"actor": "actor",
            "critic": "critic"}

# train configs
# lr: learning rate, int: interval
# warm up should be less than one epoch
ddpg_update_int = 1  # in steps
ddpg_update_batch_num = 1
ddpg_warmup_steps = 2000
model_save_int = 100  # in episodes
profile_int = 10  # in episodes


def gen_learning_rate_func(lr_map):
    def learning_rate_func(step):
        for i in range(len(lr_map) - 1):
            if lr_map[i][0] <= step < lr_map[i + 1][0]:
                return lr_map[i][1]
        return lr_map[-1][1]

    return learning_rate_func


if __name__ == "__main__":
    args = get_args()
    for k, v in args.env.items():
        globals()[k] = v
    total_steps = max_epochs * max_episodes * max_steps

    # preparations
    prep_dir_default(root_dir)
    logger.info("Directories prepared.")
    global_board.init(log_dir + "train_log")
    writer = global_board.writer

    base_actor = SwarmActor(24, 4, history_depth, len(neighbors), True, device)
    base_actor_t = SwarmActor(24, 4, history_depth, len(neighbors), True, device)
    negotiator = SwarmNegotiator(24, 4, history_depth, len(neighbors), True, device)
    negotiator_t = SwarmNegotiator(24, 4, history_depth, len(neighbors), True, device)

    # currently, all agents share the same two networks
    # TODO: implement K-Sub policies
    actor = WrappedActorNet(base_actor, negotiator)
    actor_t = WrappedActorNet(base_actor_t, negotiator_t)
    critic = WrappedCriticNet(SwarmCritic(24, 4, history_depth, len(neighbors), device))
    critic_t = WrappedCriticNet(SwarmCritic(24, 4, history_depth, len(neighbors), device))

    logger.info("Networks created")

    actor_lr_map = [[0, 1e-4],
                    [total_steps // 3, 1e-4],
                    [total_steps * 2 // 3, 1e-4],
                    [total_steps, 1e-4]]
    critic_lr_map = [[0, 1e-4],
                     [total_steps // 3, 1e-4],
                     [total_steps * 2 // 3, 1e-4],
                     [total_steps, 1e-4]]

    actor_lr_func = gen_learning_rate_func(actor_lr_map)
    critic_lr_func = gen_learning_rate_func(critic_lr_map)

    ddpg = DDPG(actor, actor_t, critic, critic_t,
                t.optim.Adam, nn.MSELoss(reduction='sum'), device,
                lr_scheduler=LambdaLR,
                lr_scheduler_params=[[actor_lr_func], [critic_lr_func]],
                replay_size=replay_size,
                batch_num=1)

    if not restart:
        ddpg.load(root_dir + "/model", save_map)
    logger.info("DDPG framework initialized")

    # training
    # preparations
    env = BipedalMultiWalker(agent_num)
    agents = [SwarmAgent(base_actor, negotiator, len(neighbors), 4, 24,
                         history_depth, mean_anneal, theta_anneal, 1, True, device)
              for i in range(agent_num)]

    # begin training
    # epoch > episode
    epoch = Counter()
    episode = Counter()
    episode_finished = False
    global_step = Counter()
    local_step = Counter()
    while epoch < max_epochs:
        epoch.count()
        logger.info("Begin epoch {}".format(epoch))
        while episode < max_episodes:
            episode.count()
            logger.info("Begin episode {}, epoch={}".format(episode, epoch))

            # environment initialization
            env.reset()
            for agent in agents:
                agent.reset()

            ### currently, agents have fixed communication topology
            for i in range(agent_num):
                agent_neighbors = []
                for j in neighbors:
                    index = i + j
                    if agent_num > index >= 0:
                        agent_neighbors.append(index)
                agents[i].set_neighbors([agents[n] for n in agent_neighbors])

            # render configuration
            if episode.get() % profile_int == 0 and global_step.get() > ddpg_warmup_steps:
                render = True
            else:
                render = False
            frames = []

            # model serialization
            if episode.get() % model_save_int == 0:
                ddpg.save(model_dir, save_map, episode.get() + (epoch.get() - 1) * max_episodes)
                logger.info("Saving model parameters, epoch={}, episode={}"
                            .format(epoch, episode))

            # batch size = 1
            episode_begin = time.time()
            actions = t.zeros([1, agent_num * 4], device=device)
            total_reward = t.zeros([1, agent_num], device=device)

            while not episode_finished and local_step.get() <= max_steps:
                global_step.count()
                local_step.count()

                step_begin = time.time()
                with t.no_grad():
                    old_samples = [agent.get_sample() for agent in agents]

                    state, reward, episode_finished, info = env.step(actions[0].to("cpu"))

                    if render:
                        frames.append(env.render(mode="rgb_array"))

                    state = t.tensor(state, dtype=t.float32, device=device)
                    reward = t.tensor(reward, dtype=t.float32, device=device).unsqueeze(dim=0)

                    total_reward += reward

                    # agent model inference
                    for ag in range(agent_num):
                        agents[ag].set_observe(state[ag * 24: (ag + 1) * 24].unsqueeze(dim=0))

                    for ag in range(agent_num):
                        agents[ag].act_step()

                    while all([agent.get_negotiation_rate() > 0.01 for agent in agents]):
                        for ag in range(agent_num):
                            agents[ag].negotiate_step()

                    for ag in range(agent_num):
                        actions[:, ag * 4: (ag + 1) * 4] = agents[ag].final_step()

                    if global_step.get() < ddpg_warmup_steps:
                        actions = ddpg.add_noise_to_action(actions,
                                                           noise_range * agent_num,
                                                           1)

                    writer.add_scalar("action_min", t.min(actions), global_step.get())
                    writer.add_scalar("action_mean", t.mean(actions), global_step.get())
                    writer.add_scalar("action_max", t.max(actions), global_step.get())

                    if local_step.get() > 1:
                        for old_sample, agent, r in zip(old_samples, agents, reward[0]):
                            sample = agent.get_sample()
                            ddpg.store_observe({"state": {"history": old_sample[0],
                                                          "observation": old_sample[1],
                                                          "neighbor_observation": old_sample[2],
                                                          "neighbor_action_all": old_sample[3],
                                                          "negotiate_rate_all": old_sample[4]},
                                                "action": {"action": agent.action},
                                                "next_state": {"history": sample[0],
                                                               "observation": sample[1],
                                                               "neighbor_observation": sample[2],
                                                               "neighbor_action_all": sample[3],
                                                               "negotiate_rate_all": sample[4]},
                                                "reward": r})
                            agent.set_reward(r)

                for agent in agents:
                    if local_step.get() > 1:
                        agent.update_history()
                    agent.reset_negotiate()
                step_end = time.time()

                writer.add_scalar("step_time", step_end - step_begin, global_step.get())
                writer.add_scalar("episodic_reward", t.mean(reward), global_step.get())
                writer.add_scalar("episodic_sum_reward", t.mean(total_reward), global_step.get())
                writer.add_scalar("episode_length", local_step.get(), global_step.get())

                logger.info("Step {} completed in {:.3f} s, epoch={}, episode={}".
                            format(local_step, step_end - step_begin, epoch, episode))

                if global_step.get() % ddpg_update_int == 0 and global_step.get() > ddpg_warmup_steps:
                    for i in range(ddpg_update_batch_num):
                        ddpg_train_begin = time.time()
                        ddpg.update(False)
                        ddpg.update_lr_scheduler()
                        ddpg_train_end = time.time()
                        logger.info("DDPG train Step {} completed in {:.3f} s, epoch={}, episode={}".
                                    format(i, ddpg_train_end - ddpg_train_begin, epoch, episode))

            if render:
                create_gif(frames, "{}/log/images/{}_{}".format(root_dir, epoch, episode))

            local_step.reset()
            episode_finished = False
            episode_end = time.time()
            logger.info("Episode {} completed in {:.3f} s, epoch={}".
                        format(episode, episode_end - episode_begin, epoch))

        episode.reset()
