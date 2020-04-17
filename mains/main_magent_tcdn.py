import time
import numpy as np
import torch as t
import torch.nn as nn
import env.magent as magent

from torch.optim.lr_scheduler import LambdaLR

from models.frameworks.ddpg import DDPG
from models.tcdn.actor import SwarmActor, WrappedActorNet
from models.tcdn.critic import SwarmCritic, WrappedCriticNet
from models.tcdn.negotiatior import SwarmNegotiator
from models.tcdn.agent import SwarmAgent

from utils.logging import default_logger as logger
from utils.tensor_board import global_board
from utils.helper_classes import Counter
from utils.prep import prep_dir_default, prep_create_dirs

# configs
restart = True
# max_batch = 8
max_epochs = 20
max_episodes = 800
max_steps = 500
theta = 1
replay_size = 100000
history_depth = 1

map_size = 20
agent_ratio = 0.01
neighbor_num = 3
# Note: id_emedding_length = 10

mean_anneal = 0.3
theta_anneal = 0.1
device = t.device("cuda:0")
root_dir = "/data/AI/tmp/multi_agent/magent/tcdn/"
model_dir = root_dir + "model/"
log_dir = root_dir + "log/"
save_map = {"actor": "actor",
            "critic": "critic"}

# train configs
# lr: learning rate, int: interval
# warm up should be less than one epoch
ddpg_update_int = 1  # in steps
ddpg_update_batch_num = 1
#ddpg_warmup_steps = 1000
ddpg_warmup_episodes = 2
model_save_int = 100  # in episodes
profile_int = 10  # in episodes


def generate_map(env, map_size, agent_ratio, left_agents_handle, right_agents_handle):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * agent_ratio
    gap = 3

    # Note: in position (x, y, 0) array, the last dimension is agent's initial direction
    # left
    n = init_num
    side = int(np.sqrt(n)) * 2
    pos = []
    for x in range(width // 2 - gap - side, width // 2 - gap - side + side, 2):
        for y in range((height - side) // 2, (height - side) // 2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(left_agents_handle, method="custom", pos=pos)

    # right
    n = init_num
    side = int(np.sqrt(n)) * 2
    pos = []
    for x in range(width // 2 + gap, width // 2 + gap + side, 2):
        for y in range((height - side) // 2, (height - side) // 2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(right_agents_handle, method="custom", pos=pos)


def generate_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    small_agent = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1,

         'step_reward': -0.005, 'kill_reward': 5, 'dead_penalty': -0.1, 'attack_penalty': -0.1,
         })

    g0 = cfg.add_group(small_agent)
    g1 = cfg.add_group(small_agent)

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    # reward shaping to encourage attack
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=0.2)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=0.2)

    return cfg


def gen_learning_rate_func(lr_map):
    def learning_rate_func(step):
        for i in range(len(lr_map) - 1):
            if lr_map[i][0] <= step < lr_map[i + 1][0]:
                return lr_map[i][1]
        return lr_map[-1][1]

    return learning_rate_func


def build_comm_network(id, pos, neighbor_num, agents):
    dist = np.sqrt(np.sum((pos - np.expand_dims(pos, axis=1))**2, axis=2))
    nearest = id[np.argsort(dist, axis=1)][:, 1:]  # exclude self, i.e. first column
    # currently, no communication range limit
    peer_num = min(nearest.shape[1], neighbor_num)
    for i in id:
        agents[i].set_neighbors([agents[nearest[i, n]] for n in range(peer_num)])


def run_agents(env, agents, group_handle, neighbor_num, is_warm_up):
    # Note: is_warm_up should only be set to False after a complete episode!
    # dead agents are deleted before running
    agent_num = len(agents)
    views, features = env.get_observation(group_handle)
    id = np.arange(agent_num, dtype=np.int32)

    build_comm_network(id, features[:, -2:], neighbor_num, agents)

    actions = np.zeros(agent_num, dtype=np.int32)
    for ag in id:
        agents[ag].set_observe(t.from_numpy(views[ag]).unsqueeze(0))

    for ag in id:
        agents[ag].act_step()

    while all([agents[ag].get_negotiation_rate() > 0.01 for ag in id]):
        for ag in id:
            agents[ag].negotiate_step()

    for ag in id:
        actions[ag] = t.argmax(agents[ag].final_step())

    if is_warm_up:
        # generate random actions
        act_dim = env.get_action_space(group_handle)[0]
        actions = np.random.randint(0, act_dim, agent_num, dtype=np.int32)

    env.set_action(group_handle, actions)


if __name__ == "__main__":
    total_steps = max_epochs * max_episodes * max_steps

    # preparations
    prep_dir_default(root_dir)
    logger.info("Directories prepared.")
    global_board.init(log_dir + "train_log")
    writer = global_board.writer

    env = magent.GridWorld(generate_config(map_size), map_size=map_size)
    agent_num = int(np.sqrt(map_size * map_size * agent_ratio)) ** 2
    group1_handle, group2_handle = env.get_handles()

    # shape: (act,)
    action_dim = env.get_action_space(group1_handle)[0]
    # shape: (view_width, view_height, n_channel)
    view_space = env.get_view_space(group1_handle)
    view_dim = np.prod(view_space)
    # shape: (ID embedding + last action + last reward + relative pos)
    feature_dim = env.get_feature_space(group1_handle)[0]

    base_actor = SwarmActor(view_dim, action_dim, history_depth, neighbor_num, False, device)
    base_actor_t = SwarmActor(view_dim, action_dim, history_depth, neighbor_num, False, device)
    negotiator = SwarmNegotiator(view_dim, action_dim, history_depth, neighbor_num, False, device)
    negotiator_t = SwarmNegotiator(view_dim, action_dim, history_depth, neighbor_num, False, device)

    # currently, all agents share the same two networks
    # TODO: implement K-Sub policies
    actor = WrappedActorNet(base_actor, negotiator)
    actor_t = WrappedActorNet(base_actor_t, negotiator_t)
    critic = WrappedCriticNet(SwarmCritic(view_dim, action_dim, history_depth, neighbor_num, device))
    critic_t = WrappedCriticNet(SwarmCritic(view_dim, action_dim, history_depth, neighbor_num, device))

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
            generate_map(env, map_size, agent_ratio, group1_handle, group2_handle)
            group1_agents = [SwarmAgent(base_actor, negotiator, neighbor_num, action_dim, view_dim,
                                        history_depth, mean_anneal, theta_anneal, 1, False, device)
                             for i in range(agent_num)]

            group2_agents = [SwarmAgent(base_actor, negotiator, neighbor_num, action_dim, view_dim,
                                        history_depth, mean_anneal, theta_anneal, 1, False, device)
                             for i in range(agent_num)]

            groups = [group1_agents, group2_agents]
            for agent in group1_agents:
                agent.reset()
            for agent in group2_agents:
                agent.reset()

            # render configuration
            if episode.get() % profile_int == 0 and episode.get() > ddpg_warmup_episodes:
                render = True
                path = log_dir + "/images/{}_{}".format(epoch, episode)
                env.set_render_dir(path)
                prep_create_dirs([path])
            else:
                render = False

            # model serialization
            if episode.get() % model_save_int == 0:
                ddpg.save(model_dir, save_map, episode.get() + (epoch.get() - 1) * max_episodes)
                logger.info("Saving model parameters, epoch={}, episode={}"
                            .format(epoch, episode))

            # batch size = 1
            episode_begin = time.time()
            total_reward = [np.zeros([agent_num], dtype=np.float),
                            np.zeros([agent_num], dtype=np.float)]

            while not episode_finished and local_step.get() <= max_steps:
                global_step.count()
                local_step.count()

                step_begin = time.time()
                with t.no_grad():
                    old_samples = [[agent.get_sample() for agent in group1_agents],
                                   [agent.get_sample() for agent in group2_agents]]

                    if episode.get() > ddpg_warmup_episodes:
                        run_agents(env, group1_agents, group1_handle, neighbor_num, False)
                        run_agents(env, group2_agents, group2_handle, neighbor_num, False)
                    else:
                        run_agents(env, group1_agents, group1_handle, neighbor_num, True)
                        run_agents(env, group2_agents, group2_handle, neighbor_num, True)

                    episode_finished = env.step()

                    # clear dead agents
                    is_alive = [env.get_alive(group1_handle), env.get_alive(group2_handle)]
                    group1_agents = [agent for agent, alive in zip(group1_agents, is_alive[0]) if alive]
                    group2_agents = [agent for agent, alive in zip(group2_agents, is_alive[1]) if alive]
                    groups = [group1_agents, group2_agents]
                    total_reward[0] = np.delete(total_reward[0], np.where(np.logical_not(is_alive[0])))
                    total_reward[1] = np.delete(total_reward[1], np.where(np.logical_not(is_alive[1])))
                    env.clear_dead()

                    if not np.all(is_alive[0]) or not np.all(is_alive[1]):
                        pass

                    reward = [env.get_reward(group1_handle), env.get_reward(group2_handle)]
                    total_reward[0] += reward[0]
                    total_reward[1] += reward[1]

                    if local_step.get() > 1:
                        for g in (0, 1):
                            for agent, old_sample, r in zip(groups[g], old_samples[g], reward[g]):
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

                for agent in group1_agents:
                    agent.reset_negotiate()
                for agent in group2_agents:
                    agent.reset_negotiate()

                step_end = time.time()

                writer.add_scalar("step_time",
                                  step_end - step_begin, global_step.get())
                writer.add_scalar("episodic_reward",
                                  np.mean(np.concatenate(reward)), global_step.get())
                writer.add_scalar("episodic_sum_reward",
                                  np.mean(np.concatenate(total_reward)), global_step.get())

                logger.info("Step {} completed in {:.3f} s, epoch={}, episode={}".
                            format(local_step, step_end - step_begin, epoch, episode))

                if render:
                    env.render()

                if global_step.get() % ddpg_update_int == 0 and episode.get() > ddpg_warmup_episodes:
                    for i in range(ddpg_update_batch_num):
                        ddpg_train_begin = time.time()
                        ddpg.update(False)
                        ddpg.update_lr_scheduler()
                        ddpg_train_end = time.time()
                        logger.info("DDPG train Step {} completed in {:.3f} s, epoch={}, episode={}".
                                    format(i, ddpg_train_end - ddpg_train_begin, epoch, episode))

            local_step.reset()
            episode_finished = False
            episode_end = time.time()
            logger.info("Episode {} completed in {:.3f} s, epoch={}".
                        format(episode, episode_end - episode_begin, epoch))

        episode.reset()
