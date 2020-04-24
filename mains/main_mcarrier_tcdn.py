import time
import torch as t
import torch.nn as nn

from models.frameworks.ddpg_td3 import DDPG_TD3
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
from utils.checker import check_model

from env.walker.carrier import BipedalMultiCarrier

# definitions
observe_dim = 28
action_dim = 4

# configs
restart = True
max_epochs = 20
max_episodes = 1000
max_steps = 2000
replay_size = 500000

agent_num = 1
history_depth = 2
neighbors = [-1]
neighbor_num = len(neighbors)
explore_noise_params = [(0, 0.2)] * action_dim
nego_mean_anneal = 0.3
nego_theta_anneal = 0.1
nego_rounds = 1
device = t.device("cuda:0")
root_dir = "/data/AI/tmp/multi_agent/walker/tcdn/"
model_dir = root_dir + "model/"
log_dir = root_dir + "log/"
save_map = {}

# train configs
# lr: learning rate, int: interval
# warm up should be less than one epoch
ddpg_update_batch_size = 100
ddpg_warmup_steps = 200
model_save_int = 100  # in episodes
profile_int = 50  # in episodes


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

    base_actor = SwarmActor(observe_dim, action_dim, history_depth, neighbor_num, True, device)
    base_actor_t = SwarmActor(observe_dim, action_dim, history_depth, neighbor_num, True, device)
    negotiator = SwarmNegotiator(observe_dim, action_dim, history_depth, neighbor_num, True, device)
    negotiator_t = SwarmNegotiator(observe_dim, action_dim, history_depth, neighbor_num, True, device)

    # currently, all agents share the same two networks
    # TODO: implement K-Sub policies
    actor = WrappedActorNet(base_actor, negotiator)
    actor_t = WrappedActorNet(base_actor_t, negotiator_t)
    critic = WrappedCriticNet(SwarmCritic(observe_dim, action_dim, history_depth, neighbor_num, device))
    critic_t = WrappedCriticNet(SwarmCritic(observe_dim, action_dim, history_depth, neighbor_num, device))
    #critic2 = WrappedCriticNet(SwarmCritic(observe_dim, action_dim, history_depth, neighbor_num, device))
    #critic2_t = WrappedCriticNet(SwarmCritic(observe_dim, action_dim, history_depth, neighbor_num, device))

    logger.info("Networks created")

    ddpg = DDPG(
                actor, actor_t, critic, critic_t, #critic2, critic2_t,
                t.optim.Adam, nn.MSELoss(reduction='sum'), device,
                discount=0.99,
                update_rate=0.005,
                learning_rate=0.001,
                replay_size=replay_size,
                batch_size=ddpg_update_batch_size)

    if not restart:
        ddpg.load(root_dir + "/model", save_map)
    logger.info("DDPG framework initialized")

    # training
    # preparations
    env = BipedalMultiCarrier(agent_num)
    agents = [SwarmAgent(base_actor, negotiator, len(neighbors), action_dim, observe_dim, history_depth,
                         mean_anneal=nego_mean_anneal, theta_anneal=nego_theta_anneal,
                         batch_size=1, contiguous=True, device=device)
              for i in range(agent_num)]

    # begin training
    # epoch > episode
    epoch = Counter()
    episode = Counter()
    episode_finished = False
    global_step = Counter()
    local_step = Counter()

    #check_model(writer, critic, global_step, name="critic")
    #check_model(writer, base_actor, global_step, name="actor")

    while epoch < max_epochs:
        epoch.count()
        logger.info("Begin epoch {}".format(epoch))
        while episode < max_episodes:
            episode.count()
            logger.info("Begin episode {}, epoch={}".format(episode, epoch))

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
            actions = t.zeros([1, agent_num * action_dim], device=device)
            total_reward = t.zeros([1, agent_num], device=device)
            state, reward = t.tensor(env.reset(), dtype=t.float32, device=device), 0
            old_samples = None

            while not episode_finished and local_step.get() <= max_steps:
                global_step.count()
                local_step.count()

                step_begin = time.time()
                with t.no_grad():

                    # agent model inference
                    for ag in range(agent_num):
                        agents[ag].set_observe(state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(dim=0))

                    for ag in range(agent_num):
                        agents[ag].act_step(local_step.get())

                    for i in range(nego_rounds):
                        for ag in range(agent_num):
                            agents[ag].negotiate_step()

                    for ag in range(agent_num):
                        actions[:, ag * action_dim: (ag + 1) * action_dim] = agents[ag].final_step()

                    if not render:
                        actions = ddpg.add_normal_noise_to_action(actions,
                                                                  explore_noise_params * agent_num,
                                                                  1)

                    actions = t.clamp(actions, min=-1, max=1)
                    state, reward, episode_finished, info = env.step(actions[0].to("cpu"))

                    if render:
                        frames.append(env.render(mode="rgb_array"))

                    state = t.tensor(state, dtype=t.float32, device=device)
                    reward = t.tensor(reward, dtype=t.float32, device=device).unsqueeze(dim=0)

                    total_reward += reward

                    for agent, r in zip(agents, reward[0]):
                        agent.set_reward(r.view(1, 1))

                    if old_samples is not None:
                        for old_sample, agent, r in zip(old_samples, agents, reward[0]):
                            sample = agent.get_sample()
                            ddpg.store_observe({"state": {"history": old_sample[0],
                                                          "history_time_steps": old_sample[1],
                                                          "observation": old_sample[2],
                                                          "neighbor_observation": old_sample[3],
                                                          "neighbor_action_all": old_sample[4],
                                                          "negotiate_rate_all": old_sample[5],
                                                          "time_step": old_sample[6]},
                                                "action": {"action": agent.action},
                                                "next_state": {"history": sample[0],
                                                               "history_time_steps": sample[1],
                                                               "observation": sample[2],
                                                               "neighbor_observation": sample[3],
                                                               "neighbor_action_all": sample[4],
                                                               "negotiate_rate_all": sample[5],
                                                               "time_step": sample[6]},
                                                "reward": float(r),
                                                "terminal": episode_finished or local_step.get() == max_steps})


                    writer.add_scalar("action_min", t.min(actions), global_step.get())
                    writer.add_scalar("action_mean", t.mean(actions), global_step.get())
                    writer.add_scalar("action_max", t.max(actions), global_step.get())

                    old_samples = [agent.get_sample() for agent in agents]

                for agent in agents:
                    agent.update_history(local_step.get())
                    agent.reset_negotiate()
                step_end = time.time()

                writer.add_scalar("step_time", step_end - step_begin, global_step.get())
                writer.add_scalar("episodic_reward", t.mean(reward), global_step.get())
                writer.add_scalar("episodic_sum_reward", t.mean(total_reward), global_step.get())
                writer.add_scalar("episode_length", local_step.get(), global_step.get())

                logger.info("Step {} completed in {:.3f} s, epoch={}, episode={}".
                            format(local_step, step_end - step_begin, epoch, episode))

            logger.info("Sum reward: {}, epoch={}, episode={}".format(
                t.mean(total_reward), epoch, episode))

            if global_step.get() > ddpg_warmup_steps:
                for i in range(local_step.get()):
                    ddpg_train_begin = time.time()
                    # if using non-batched agents, set concatenate_samples=False
                    ddpg.update(update_policy=i % 2 == 0, update_targets=i % 2 == 0)
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
