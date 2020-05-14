import itertools as it
import torch as t
import torch.nn as nn
from datetime import datetime as dt

from models.models.base import StaticModuleWrapper as MW
from models.frameworks.hddpg import HDDPG
from models.naive.env_magent_ddpg import Actor, Critic

from utils.logging import default_logger as logger
from utils.tensor_board import global_board
from utils.helper_classes import Counter, Timer, Object
from utils.conf import Config
from utils.save_env import SaveEnv
from utils.prep import prep_args

from env.magent_helper import *

from .utils import draw_agent_num_figure

# configs
c = Config()

c.map_size = 50
c.agent_ratio = 0.04
c.neighbor_num = 3
agent_num = int(np.sqrt(c.map_size * c.map_size * c.agent_ratio)) ** 2

#c.restart_from_trial = "2020_05_06_21_50_57"
c.max_episodes = 20000
c.max_steps = 2000
c.replay_size = 500000

c.agent_num = 3
c.q_increase_rate = 1
c.q_decrease_rate = 1
c.device = "cuda:0"
c.storage_device = "cpu"
c.root_dir = "/data/AI/tmp/multi_agent/mcarrier/hdqn/"

# train configs
# lr: learning rate, int: interval
# warm up should be less than one epoch
c.ddpg_update_batch_size = 100
c.ddpg_warmup_steps = 200
c.model_save_int = 100  # in episodes
c.profile_int = 50  # in episodes


def run_agents(env, ddpg, group_handle, add_noise=False):
    # dead agents are deleted before running
    views, features = env.get_observation(group_handle)
    real_num = views.shape[0]

    views = np.transpose(views, [0, 3, 1, 2])

    views = t.from_numpy(views).to(c.device)
    features = t.from_numpy(features).to(c.device)

    if add_noise:
        actions = ddpg.act_discreet_with_noise({"view": views, "feature": features})
    else:
        actions = ddpg.act_discreet({"view": views, "feature": features})

    env.set_action(group_handle, actions.flatten().to("cpu").numpy().astype(np.int32))

    return real_num, actions, \
           views.to(c.storage_device), \
           features.to(c.storage_device)


def create_models():
    config = generate_combat_config(c.map_size)
    view_shape, feature_shape, action_dim = get_agent_io_shapes(config, c.map_size)[0]

    actor = MW(Actor(view_shape, feature_shape,
                     action_dim, c.conv).to(c.device), c.device, c.device)
    actor_t = MW(Actor(view_shape, feature_shape,
                       action_dim, c.conv).to(c.device), c.device, c.device)
    critic = MW(Critic(view_shape, feature_shape,
                       c.conv).to(c.device), c.device, c.device)
    critic_t = MW(Critic(view_shape, feature_shape,
                         c.conv).to(c.device), c.device, c.device)

    ddpg = HDDPG(actor, actor_t, critic, critic_t,
                 t.optim.Adam, nn.MSELoss(reduction='sum'),
                 q_increase_rate=c.q_increase_rate,
                 q_decrease_rate=c.q_decrease_rate,
                 discount=0.99,
                 update_rate=0.005,
                 batch_size=c.ddpg_update_batch_size,
                 learning_rate=0.001,
                 replay_size=c.replay_size,
                 replay_device=c.storage_device)

    if c.restart_from_trial is not None:
        ddpg.load(save_env.get_trial_model_dir())

    return ddpg


if __name__ == "__main__":
    save_env = SaveEnv(c.root_dir, restart_use_trial=c.restart_from_trial)
    prep_args(c, save_env)

    # save_env.remove_trials_older_than(diff_hour=1)
    global_board.init(save_env.get_trial_train_log_dir())
    writer = global_board.writer
    logger.info("Directories prepared.")

    ddpg = create_models()
    logger.info("DDPG framework initialized")

    # training
    # preparations
    config = generate_combat_config(c.map_size)
    env = magent.GridWorld(config, map_size=c.map_size)
    env.reset()

    # begin training
    episode = Counter()
    episode_finished = False
    global_step = Counter()
    local_step = Counter()
    timer = Timer()

    while episode < c.max_episodes:
        episode.count()
        logger.info("Begin episode {} at {}".format(episode, dt.now().strftime("%m/%d-%H:%M:%S")))

        # environment initialization
        env.reset()

        group_handles = env.get_handles()
        generate_combat_map(env, c.map_size, c.agent_ratio, group_handles[0], group_handles[1])

        # render configuration
        if episode.get() % c.profile_int == 0:
            path = save_env.get_trial_image_dir() + "/{}".format(episode)
            save_env.create_dirs([path])
            env.set_render_dir(path)
            render = True
        else:
            render = False

        # model serialization
        if episode.get() % c.model_save_int == 0:
            ddpg.save(save_env.get_trial_model_dir(), version=episode.get())
            logger.info("Saving model parameters, episode={}".format(episode))

        # batch size = 1
        total_reward = [0, 0]
        agent_alive_ids = [[ag for ag in range(agent_num)] for _ in (0, 1)]
        agent_dead_ids = [[] for _ in (0, 1)]
        agent_alive_history = [[] for _ in (0, 1)]
        agent_real_nums = [None, None]
        tmp_observes = [[[] for _ in range(agent_num)] for __ in (0, 1)]

        local_step = Counter()
        episode_finished = False

        while not episode_finished and local_step.get() <= c.max_steps:
            global_step.count()
            local_step.count()

            timer.begin()
            with t.no_grad():
                agent_status = [Object(), Object()]
                for g in (0, 1):
                    agent_real_nums[g], agent_status[g].actions, \
                    agent_status[g].views, agent_status[g].features = \
                        run_agents(env, ddpg, group_handles[g], True)

                episode_finished = env.step()
                # reward and is_alive must be get before clear_dead() !
                reward = [env.get_reward(h) for h in group_handles]
                is_alive = [env.get_alive(h) for h in group_handles]

                for g in (0, 1):
                    # remove dead ids
                    agent_alive_ids[g] = [id for id, is_alive in
                                          zip(agent_alive_ids[g], is_alive[g])
                                          if is_alive]
                    agent_dead_ids[g] += [id for id, is_alive in
                                          zip(agent_alive_ids[g], is_alive[g])
                                          if not is_alive]

                agent_alive_history[0].append(np.sum(is_alive[0]))
                agent_alive_history[1].append(np.sum(is_alive[1]))

                total_reward[0] += np.mean(reward[0])
                total_reward[1] += np.mean(reward[1])

                if render:
                    env.render()

                if local_step.get() > 1:
                    for g in (0, 1):
                        for aid, idx in zip(agent_alive_ids[g], range(agent_real_nums[g])):
                            status = agent_status[g]
                            tmp_observes[g][aid].append(
                                {"state": {"view": status.views[idx].unsqueeze(0).clone(),
                                           "feature": status.features[idx].unsqueeze(0).clone()},
                                 "action": {"action": status.actions[idx].unsqueeze(0).clone()},
                                 "next_state": {},
                                 "reward": float(reward[g][idx]),
                                 "terminal": episode_finished or local_step.get() == c.max_steps,
                                 }
                            )
                        for aid in agent_dead_ids[g]:
                            tmp_observes[g][aid][-1]["terminal"] = True

                env.clear_dead()

        for g in (0, 1):
            for ag in range(agent_num):
                tmp_observe = tmp_observes[g][ag]
                for i in reversed(range(1, len(tmp_observe))):
                    tmp_observe[i - 1]["next_state"] = tmp_observe[i]["state"]

                for record in tmp_observe:
                    ddpg.store_observe(record)

        logger.info("Sum reward: {}, episode={}".format(total_reward, episode))
        writer.add_scalar("episodic_g1_sum_reward", total_reward[0], episode.get())
        writer.add_scalar("episodic_g2_sum_reward", total_reward[1], episode.get())
        writer.add_figure("agent_num", draw_agent_num_figure(agent_real_nums), episode.get())
        writer.add_scalar("episode_length", local_step, episode.get())

        if global_step.get() > c.ddpg_warmup_steps:
            for i in range(local_step.get()):
                timer.begin()
                ddpg.update(update_policy=i % 2 == 0, update_targets=i % 2 == 0)
                ddpg.update_lr_scheduler()
                writer.add_scalar("train_step_time", timer.end(), global_step.get())

        local_step.reset()
        episode_finished = False
        logger.info("End episode {} at {}".format(episode, dt.now().strftime("%m/%d-%H:%M:%S")))
