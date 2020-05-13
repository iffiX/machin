import time
import torch as t
import torch.nn as nn
import numpy as np

from env.magent_helper import *

from models.frameworks.hddpg import HDDPG
from models.naive.env_walker import Actor, Critic

from utils.logging import default_logger as logger
from utils.image import create_gif
from utils.tensor_board import global_board
from utils.helper_classes import Counter
from utils.prep import prep_dirs_default, prep_create_dirs
from utils.args import get_args

from env.walker.carrier import BipedalMultiCarrier

# configs
restart = True
max_epochs = 20
max_episodes = 1000
max_steps = 2000
replay_size = 500000

map_size = 20
agent_ratio = 0.01
neighbor_num = 3

#explore_noise_params = (0, 0.2)
explore_noise_params = (0, 1)
q_increase_rate = 1
q_decrease_rate = 1
device = t.device("cuda:0")
root_dir = "/data/AI/tmp/multi_agent/magent/hdqn/"
model_dir = root_dir + "model/"
log_dir = root_dir + "log/"
save_map = {}

# train configs
# lr: learning rate, int: interval
# warm up should be less than one epoch
ddpg_update_batch_size = 256
ddpg_warmup_steps = 200
model_save_int = 10  # in episodes
profile_int = 10  # in episodes


def run_agents(env, ddpg, group_handle, agent_num, neighbor_num, add_noise):
    # Note: is_warm_up should only be set to False after a complete episode!
    # dead agents are deleted before running
    views, features = env.get_observation(group_handle)
    id = np.arange(agent_num, dtype=np.int32)

    views = t.from_numpy(views).to(ddpg.device).flatten(1, -1)
    raw_actions = []
    actions = np.zeros(agent_num, dtype=np.int32)
    for ag in id:
        state = views[ag].unsqueeze(0)
        if add_noise:
            action = t.softmax(ddpg.act_with_noise({"state": state}, explore_noise_params, mode="normal"),
                               dim=1)
            raw_actions.append(action)
            actions[ag] = t.argmax(action)
        else:
            action = ddpg.act({"state": state})
            raw_actions.append(action)
            actions[ag] = t.argmax(action)
    print(actions)
    env.set_action(group_handle, actions)
    return t.cat(raw_actions, dim=0), actions, views


if __name__ == "__main__":
    args = get_args()
    for k, v in args.env.items():
        globals()[k] = v
    total_steps = max_epochs * max_episodes * max_steps

    # preparations
    prep_dirs_default(root_dir)
    logger.info("Directories prepared.")
    global_board.init(log_dir + "train_log")
    writer = global_board.writer

    env = magent.GridWorld(generate_combat_config(map_size), map_size=map_size)
    agent_num = int(np.sqrt(map_size * map_size * agent_ratio)) ** 2
    group1_handle, group2_handle = env.get_handles()

    # shape: (act,)
    action_dim = env.get_action_space(group1_handle)[0]
    # shape: (view_width, view_height, n_channel)
    observe_space = env.get_view_space(group1_handle)
    observe_dim = np.prod(observe_space)
    # shape: (ID embedding + last action + last reward + relative pos)
    feature_dim = env.get_feature_space(group1_handle)[0]

    actor = Actor(observe_dim, action_dim, 1).to(device)
    actor_t = Actor(observe_dim, action_dim, 1).to(device)
    critic = Critic(observe_dim, action_dim).to(device)
    critic_t = Critic(observe_dim, action_dim).to(device)

    logger.info("Networks created")

    ddpg = HDDPG(
                actor, actor_t, critic, critic_t,
                t.optim.Adam, nn.MSELoss(reduction='sum'), device,
                q_increase_rate=q_increase_rate,
                q_decrease_rate=q_decrease_rate,
                discount=0.99,
                update_rate=0.005,
                batch_size=ddpg_update_batch_size,
                learning_rate=0.001,
                replay_size=replay_size)

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
            generate_combat_map(env, map_size, agent_ratio, group1_handle, group2_handle)

            # render configuration
            if episode.get() % profile_int == 0 and global_step.get() > ddpg_warmup_steps:
                render = True
                path = log_dir + "/images/{}_{}_{}".format(epoch, episode, global_step)
                env.set_render_dir(path)
                prep_create_dirs([path])
            else:
                render = False

            # model serialization
            if episode.get() % model_save_int == 0:
                ddpg.save(model_dir, save_map, global_step.get())
                logger.info("Saving model parameters, epoch={}, episode={}"
                            .format(epoch, episode))

            # batch size = 1
            episode_begin = time.time()
            total_reward = [np.zeros([agent_num], dtype=np.float),
                            np.zeros([agent_num], dtype=np.float)]
            old_states, states = [None, None], [None, None]
            actions, raw_actions = [None, None], [None, None]

            while not episode_finished and local_step.get() <= max_steps:
                old_states = states
                global_step.count()
                local_step.count()

                step_begin = time.time()
                with t.no_grad():
                    current_agent_g1_num = total_reward[0].shape[0]
                    current_agent_g2_num = total_reward[1].shape[0]

                    # agent model inference
                    if render:
                        raw_actions[0], actions[0], states[0] = \
                            run_agents(env, ddpg, group1_handle, current_agent_g1_num, neighbor_num, False)
                        raw_actions[1], actions[1], states[1] = \
                            run_agents(env, ddpg, group2_handle, current_agent_g2_num, neighbor_num, False)
                    else:
                        raw_actions[0], actions[0], states[0] = \
                            run_agents(env, ddpg, group1_handle, current_agent_g1_num, neighbor_num, True)
                        raw_actions[1], actions[1], states[1] = \
                            run_agents(env, ddpg, group2_handle, current_agent_g2_num, neighbor_num, True)

                    episode_finished = env.step()

                    # clear dead agents
                    is_alive = [env.get_alive(group1_handle), env.get_alive(group2_handle)]
                    total_reward[0] = np.delete(total_reward[0], np.where(np.logical_not(is_alive[0])))
                    total_reward[1] = np.delete(total_reward[1], np.where(np.logical_not(is_alive[1])))
                    env.clear_dead()

                    reward = [env.get_reward(group1_handle), env.get_reward(group2_handle)]
                    total_reward[0] += reward[0]
                    total_reward[1] += reward[1]

                    if old_states[0] is not None and old_states[1] is not None:
                        for ag in range(agent_num):
                            for g in (0, 1):
                                for old_st, act, st, r in zip(old_states[g], raw_actions[g], states[g], reward[g]):
                                    ddpg.store_observe({"state": {"state": old_st.unsqueeze(0).clone()},
                                                        "action": {"action": act.unsqueeze(0).clone()},
                                                        "next_state": {"state": st.unsqueeze(0).clone()},
                                                        "reward": r,
                                                        "terminal": episode_finished or local_step.get() == max_steps})

                    writer.add_histogram("action_group1", actions[0], global_step.get())
                    writer.add_histogram("action_group2", actions[1], global_step.get())

                step_end = time.time()

                writer.add_scalar("step_time", step_end - step_begin, global_step.get())
                writer.add_scalar("group1_episodic_reward", np.mean(reward[0]), global_step.get())
                writer.add_scalar("group1_episodic_sum_reward", np.mean(total_reward[0]), global_step.get())
                writer.add_scalar("group2_episodic_reward", np.mean(reward[1]), global_step.get())
                writer.add_scalar("group2_episodic_sum_reward", np.mean(total_reward[1]), global_step.get())
                writer.add_scalar("episode_length", local_step.get(), global_step.get())

                logger.info("Step {} completed in {:.3f} s, epoch={}, episode={}".
                            format(local_step, step_end - step_begin, epoch, episode))

            if global_step.get() > ddpg_warmup_steps:
                for i in range(local_step.get()):
                    ddpg_train_begin = time.time()
                    ddpg.update(update_policy=i % 2 == 0, update_targets=i % 2 == 0)
                    ddpg_train_end = time.time()
                    logger.info("DDPG train Step {} completed in {:.3f} s, epoch={}, episode={}".
                                format(i, ddpg_train_end - ddpg_train_begin, epoch, episode, global_step.get()))

            if render:
                env.render()

            local_step.reset()
            episode_finished = False
            episode_end = time.time()
            logger.info("Episode {} completed in {:.3f} s, epoch={}".
                        format(episode, episode_end - episode_begin, epoch))

        episode.reset()
