import torch as t
import torch.nn as nn

from datetime import datetime as dt

from models.models.base import StaticModuleWrapper as MW
from models.frameworks.ppo import PPO
from models.naive.env_magent_ppo import Actor, Critic

from utils.logging import default_logger as logger
from utils.image import create_gif
from utils.tensor_board import global_board
from utils.helper_classes import Counter, Timer
from utils.conf import Config
from utils.save_env import SaveEnv
from utils.prep import prep_args
from utils.parallel import get_context, Pool

from env.magent_helper import *

# configs
c = Config()

c.map_size = 20
c.agent_ratio = 0.01
c.neighbor_num = 3

# c.restart_from_trial = "2020_05_09_15_00_31"
c.max_episodes = 5000
c.max_steps = 500
c.replay_size = 10000

c.device = "cuda:0"
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
c.ppo_update_int = 5  # = the number of episodes stored in ppo replay buffer
c.model_save_int = c.ppo_update_int * 20  # in episodes
c.profile_int = 50  # in episodes


def run_agents(env, ppo, group_handle, agent_num, neighbor_num, add_noise):
    # dead agents are deleted before running
    views, features = env.get_observation(group_handle)
    id = np.arange(agent_num, dtype=np.int32)

    views = t.from_numpy(views).to(c.device).flatten(start_dim=1)
    features = t.from_numpy(features).to(c.device)

    actions, probs, *_ = ppo.act({"view": views, "feature": features})

    env.set_action(group_handle, actions.flatten().to("cpu").numpy())
    return actions, probs, views, features


if __name__ == "__main__":
    save_env = SaveEnv(c.root_dir, restart_use_trial=c.restart_from_trial)
    prep_args(c, save_env)

    # save_env.remove_trials_older_than(diff_hour=1)
    global_board.init(save_env.get_trial_train_log_dir())
    writer = global_board.writer
    logger.info("Directories prepared.")

    agent_num = int(np.sqrt(c.map_size * c.map_size * c.agent_ratio)) ** 2
    config = generate_combat_config(c.map_size)
    view_shape, feature_shape, action_dim = get_agent_io_shapes(config, c.map_size)[0]

    actor = MW(Actor(view_shape, feature_shape,
                     action_dim, c.conv).to(c.device), c.device, c.device)
    critic = MW(Critic(view_shape, feature_shape,
                       c.conv).to(c.device), c.device, c.device)

    actor.share_memory()
    critic.share_memory()
    logger.info("Networks created")

    # default replay buffer storage is main cpu mem
    # when stored in main mem, takes about 0.65e-3 sec to move result from gpu to cpu,
    ppo = PPO(actor, critic,
              t.optim.Adam, nn.MSELoss(reduction='sum'),
              replay_device=c.device,
              replay_size=c.replay_size,
              entropy_weight=c.entropy_weight,
              discount=c.discount,
              update_times=c.ppo_update_times,
              batch_size=c.ppo_update_batch_size,
              learning_rate=c.learning_rate)

    if c.restart_from_trial is not None:
        ppo.load(save_env.get_trial_model_dir())
    logger.info("PPO framework initialized")

    # training
    # preparations
    ctx = get_context("spawn")
    pool = Pool(processes=c.workers, context=ctx)
    pool.enable_global_find(True)

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
            env = magent.GridWorld(config, map_size=c.map_size)
            group1_handle, group2_handle = env.get_handles()
            generate_combat_map(env, c.map_size, c.agent_ratio, group1_handle, group2_handle)

            # render configuration
            if episode_num % c.profile_int == 0:
                render = True
                path = save_env.get_trial_image_dir() + "{}".format(episode)
                save_env.crea
                env.set_render_dir(path)
            else:
                render = False
            frames = []

            # batch size = 1
            total_reward = 0
            state, reward = t.tensor(env.reset(), dtype=t.float32, device=c.device), 0

            tmp_observe = []
            local_step = Counter()
            episode_finished = False

            while not episode_finished and local_step.get() <= c.max_steps:
                local_step.count()
                timer.begin()
                with t.no_grad():
                    old_state = state

                    # agent model inference
                    action, prob, *_ = ppo.act({"state": state.unsqueeze(0)})

                    state, reward, episode_finished, _ = env.step(action[0].to("cpu"))

                    if render:
                        frames.append(env.render(mode="rgb_array"))

                    state = t.tensor(state, dtype=t.float32, device=c.device)

                    total_reward += reward

                    tmp_observe.append({"state": {"state": old_state.unsqueeze(0).clone()},
                                        "action": {"action": action.clone()},
                                        "next_state": {"state": state.unsqueeze(0).clone()},
                                        "reward": float(reward),
                                        "terminal": episode_finished or local_step.get() == c.max_steps,
                                        "action_log_prob": float(prob)
                                        })

            # ordinary sampling, calculate value for each observation
            tmp_observe[-1]["value"] = tmp_observe[-1]["reward"]
            for i in reversed(range(1, len(tmp_observe))):
                tmp_observe[i - 1]["value"] = \
                    tmp_observe[i]["value"] * c.discount + tmp_observe[i - 1]["reward"]

            return tmp_observe, total_reward, local_step.get(), frames


        results = pool.map(run_trial, range(first_episode, last_episode + 1))

        for result, episode_num in zip(results, range(first_episode, last_episode + 1)):
            tmp_observe, total_reward, local_step, frames = result
            logger.info("Sum reward: {}, episode={}".format(float(total_reward), episode_num))
            writer.add_scalar("episodic_sum_reward", float(total_reward), episode_num)
            writer.add_scalar("episode_length", local_step, episode_num)

            for obsrv in tmp_observe:
                ppo.store_observe(obsrv)

            if len(frames) != 0:
                # sub-processes cannot start a sub-process
                # so we have to store results in the main process
                create_gif(frames, save_env.get_trial_image_dir() + "/{}".format(episode_num))

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
