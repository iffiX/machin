import itertools as it
import torch as t
import torch.nn as nn
from datetime import datetime as dt

from models.models.base import StaticModuleWrapper as MW
from models.frameworks.ppo import PPO
from models.naive.env_magent_ppo import Actor, Critic

from utils.logging import default_logger as logger
from utils.tensor_board import global_board
from utils.helper_classes import Counter, Timer, Object
from utils.conf import Config
from utils.save_env import SaveEnv
from utils.prep import prep_args
from utils.parallel import get_context, Pool, mark_static_module

from env.magent_helper import *

from .utils import draw_agent_num_figure

mark_static_module(magent)

# configs
c = Config()

c.map_size = 50
c.agent_ratio = 0.04
agent_num = int(np.sqrt(c.map_size * c.map_size * c.agent_ratio)) ** 2
c.neighbor_num = 3

# c.restart_from_trial = "2020_05_09_15_00_31"
c.max_episodes = 5000
c.max_steps = 500
c.replay_size = 20000

c.device = "cuda:0"
c.storage_device = "cpu"
c.root_dir = "/data/AI/tmp/multi_agent/magent/naive_ppo_parallel/"

# train configs
# lr: learning rate, int: interval
c.conv = True
c.workers = 2
c.discount = 0.99
c.learning_rate = 1e-3
c.entropy_weight = None
c.ppo_update_batch_size = 100
c.ppo_update_times = 50
c.ppo_update_int = 2  # = the number of episodes stored in ppo replay buffer
c.model_save_int = c.ppo_update_int * 20  # in episodes
c.profile_int = 50  # in episodes


def run_agents(env, ppo, group_handle):
    # dead agents are deleted before running
    views, features = env.get_observation(group_handle)
    real_num = views.shape[0]

    views = np.transpose(views, [0, 3, 1, 2])

    views = t.from_numpy(views).to(c.device)
    features = t.from_numpy(features).to(c.device)

    actions, probs, *_ = ppo.act({"view": views, "feature": features})

    env.set_action(group_handle, actions.flatten().to("cpu").numpy().astype(np.int32))
    return real_num, \
           actions.to(c.storage_device), \
           probs.to(c.storage_device), \
           views.to(c.storage_device), \
           features.to(c.storage_device)


def create_models():
    config = generate_combat_config(c.map_size)
    view_shape, feature_shape, action_dim = get_agent_io_shapes(config, c.map_size)[0]

    actor = MW(Actor(view_shape, feature_shape,
                     action_dim, c.conv).to(c.device), c.device, c.device)
    critic = MW(Critic(view_shape, feature_shape,
                       c.conv).to(c.device), c.device, c.device)

    actor.share_memory()
    critic.share_memory()

    ppo = PPO(actor, critic,
              t.optim.Adam, nn.MSELoss(reduction='sum'),
              replay_device=c.storage_device,
              replay_size=c.replay_size,
              entropy_weight=c.entropy_weight,
              discount=c.discount,
              update_times=c.ppo_update_times,
              batch_size=c.ppo_update_batch_size,
              learning_rate=c.learning_rate)

    if c.restart_from_trial is not None:
        ppo.load(save_env.get_trial_model_dir())

    return ppo


if __name__ == "__main__":
    save_env = SaveEnv(c.root_dir, restart_use_trial=c.restart_from_trial)
    prep_args(c, save_env)

    # save_env.remove_trials_older_than(diff_hour=1)
    global_board.init(save_env.get_trial_train_log_dir())
    writer = global_board.writer
    logger.info("Directories prepared.")

    ppo = create_models()
    logger.info("PPO framework initialized")

    # training
    # preparations
    ctx = get_context("spawn")
    pool = Pool(processes=c.workers, context=ctx)
    pool.enable_global_find(True)
    pool.enable_copy_tensors(False)

    # begin training
    episode = Counter(step=c.ppo_update_int)
    timer = Timer()

    while episode < c.max_episodes:
        first_episode = episode.get()
        episode.count()
        last_episode = episode.get() - 1
        logger.info("Begin episode {}-{} at {}".format(first_episode, last_episode,
                                                       dt.now().strftime("%m/%d-%H:%M:%S")))

        # begin trials
        def run_trial(episode_num):
            config = generate_combat_config(c.map_size)
            env = magent.GridWorld(config, map_size=c.map_size)
            env.reset()

            group_handles = env.get_handles()
            generate_combat_map(env, c.map_size, c.agent_ratio, group_handles[0], group_handles[1])

            # render configuration
            if episode_num % c.profile_int == 0:
                path = save_env.get_trial_image_dir() + "/{}".format(episode)
                save_env.create_dirs([path])
                env.set_render_dir(path)
                render = True
            else:
                render = False

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
                local_step.count()
                timer.begin()

                with t.no_grad():
                    agent_status = [Object(), Object()]
                    for g in (0, 1):
                        agent_real_nums[g], agent_status[g].actions, agent_status[g].probs, \
                        agent_status[g].views, agent_status[g].features = \
                            run_agents(env, ppo, group_handles[g])

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
                                     "action_log_prob": float(status.probs[idx])
                                     }
                                )
                            for aid in agent_dead_ids[g]:
                                tmp_observes[g][aid][-1]["terminal"] = True

                    env.clear_dead()


            # ordinary sampling, calculate value for each observation
            for g in (0, 1):
                for ag in range(agent_num):
                    tmp_observe = tmp_observes[g][ag]
                    tmp_observe[-1]["value"] = tmp_observe[-1]["reward"]
                    for i in reversed(range(1, len(tmp_observe))):
                        tmp_observe[i - 1]["value"] = \
                            tmp_observe[i]["value"] * c.discount + tmp_observe[i - 1]["reward"]

            tmp_observes = [tmp_observes[g][ag] for g in (0, 1) for ag in range(agent_num)]

            return list(it.chain(*tmp_observes))[:int(c.replay_size / c.ppo_update_int)], \
                   total_reward, local_step.get(), agent_alive_history


        results = pool.map(run_trial, range(first_episode, last_episode + 1))

        for result, episode_num in zip(results, range(first_episode, last_episode + 1)):
            tmp_observe, total_reward, local_step, agent_real_nums = result
            logger.info("Sum reward: {}, episode={}".format(total_reward, episode_num))
            writer.add_scalar("episodic_g1_sum_reward", total_reward[0], episode_num)
            writer.add_scalar("episodic_g2_sum_reward", total_reward[1], episode_num)
            writer.add_figure("agent_num", draw_agent_num_figure(agent_real_nums), episode_num)
            writer.add_scalar("episode_length", local_step, episode_num)

            for obsrv in tmp_observe:
                ppo.store_observe(obsrv)

            # model serialization
            if episode_num % c.model_save_int == 0:
                ppo.save(save_env.get_trial_model_dir(), version=episode_num)

        logger.info("End episode {}-{} at {}".format(first_episode, last_episode,
                                                     dt.now().strftime("%m/%d-%H:%M:%S")))

        # begin training
        timer.begin()
        ppo.update()
        ppo.update_lr_scheduler()
        writer.add_scalar("train_step_time", timer.end(), episode.get())

        logger.info("Train end, time = {:.2f} s, episode={}".format(timer.end(), episode))
