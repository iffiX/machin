import torch as t
import torch.nn as nn
from datetime import datetime as dt

from models.nets.base import StaticModuleWrapper as MW
from models.frames.algorithms.maddpg import MADDPG
from models.naive.env_mcarrier_maddpg import Actor, Critic

from utils.logging import default_logger as logger
from utils.image import create_gif_subproc
from utils.tensor_board import global_board
from utils.helper_classes import Counter, Timer
from utils.conf import Config
from utils.save_env import SaveEnv
from utils.prep import prep_args

from env.walker.carrier import BipedalMultiCarrier

# definitions
observe_dim = 28
action_dim = 4

# configs
c = Config()
# c.restart_from_trial = "2020_05_06_21_50_57"
c.max_episodes = 20000
c.max_steps = 2000
c.replay_size = 500000

c.agent_num = 3
c.sub_policy_num = 1
c.explore_noise_params = (0, 0.2)
c.q_increase_rate = 1
c.q_decrease_rate = 1
c.device = "cuda:0"
c.root_dir = "/data/AI/tmp/multi_agent/mcarrier/maddpg/"

# train configs
# lr: learning rate, int: interval
# warm up should be less than one epoch
c.ddpg_update_batch_size = 100
c.ddpg_warmup_steps = 2000
c.ddpg_average_target_int = 200
c.model_save_int = 100  # in episodes
c.profile_int = 50  # in episodes


if __name__ == "__main__":
    save_env = SaveEnv(c.root_dir, restart_use_trial=c.restart_from_trial)
    prep_args(c, save_env)

    # save_env.remove_trials_older_than(diff_hour=1)
    global_board.init(save_env.get_trial_train_log_dir())
    writer = global_board.writer
    logger.info("Directories prepared.")

    actor = MW(Actor(observe_dim, action_dim, 1))
    actor_t = MW(Actor(observe_dim, action_dim, 1))
    critic = MW(Critic(c.agent_num, observe_dim, action_dim))
    critic_t = MW(Critic(c.agent_num, observe_dim, action_dim))
    logger.info("Networks created")

    ddpg = MADDPG(c.agent_num, actor, actor_t, critic, critic_t,
                  t.optim.Adam, nn.MSELoss(reduction='sum'),
                  sub_policy_num=c.sub_policy_num,
                  discount=0.99,
                  update_rate=0.005,
                  available_devices=["cuda:0"],
                  batch_size=c.ddpg_update_batch_size,
                  learning_rate=1e-3,
                  replay_size=c.replay_size,
                  replay_device="cpu")

    if c.restart_from_trial is not None:
        ddpg.load(save_env.get_trial_model_dir())
    logger.info("DDPG framework initialized")

    # training
    # preparations
    env = BipedalMultiCarrier(agent_num=c.agent_num)

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

        # render configuration
        if episode.get() % c.profile_int == 0 and global_step.get() > c.ddpg_warmup_steps:
            render = True
        else:
            render = False
        frames = []

        # model serialization
        if episode.get() % c.model_save_int == 0:
            ddpg.save(save_env.get_trial_model_dir(), version=episode.get())
            logger.info("Saving model parameters, episode={}".format(episode))

        # batch size = 1
        actions = t.zeros([1, c.agent_num, action_dim], device=c.device)
        total_reward = t.zeros([1, c.agent_num], device=c.device)
        state, reward = t.tensor(env.reset(), dtype=t.float32, device=c.device)\
                            .view([1, c.agent_num, -1]), 0

        while not episode_finished and local_step.get() <= c.max_steps:
            global_step.count()
            local_step.count()

            timer.begin()
            with t.no_grad():
                old_state = state

                # agent model inference
                if not render:
                    actions[0] = ddpg.act_with_noise(
                        {"state": state.flatten(0, 1)},
                        c.explore_noise_params, mode="normal"
                    )
                else:
                    actions[0] = ddpg.act({"state": state.flatten(0, 1)})

                actions = t.clamp(actions, min=-1, max=1)
                state, reward, episode_finished, _ = env.step(t.flatten(actions.to("cpu")))

                if render:
                    frames.append(env.render(mode="rgb_array"))

                state = t.tensor(state, dtype=t.float32, device=c.device).view([1, c.agent_num, -1])
                reward = t.tensor(reward, dtype=t.float32, device=c.device).unsqueeze(dim=0)

                total_reward += reward

                for ag in range(c.agent_num):
                    ddpg.store_transition({"state": {"state": old_state[:, ag].clone(),
                                                  "all_states": old_state.clone()},
                                        "action": {"all_actions": actions.clone()},
                                        "next_state": {"state": state[:, ag].clone(),
                                                       "all_states": state.clone()},
                                        "reward": float(reward[0][ag]),
                                        "index": ag,
                                           "terminal": episode_finished or local_step.get() == c.max_steps})

                writer.add_scalar("action_min", t.min(actions), global_step.get())
                writer.add_scalar("action_mean", t.mean(actions), global_step.get())
                writer.add_scalar("action_max", t.max(actions), global_step.get())

            writer.add_scalar("step_time", timer.end(), global_step.get())
            writer.add_scalar("episodic_reward", t.mean(reward), global_step.get())
            writer.add_scalar("episodic_sum_reward", t.mean(total_reward), global_step.get())
            writer.add_scalar("episode_length", local_step.get(), global_step.get())

        logger.info("Sum reward: {}, episode={}".format(total_reward, episode))

        if global_step.get() > c.ddpg_warmup_steps:
            for i in range(local_step.get()):
                timer.begin()
                ddpg.update(update_policy=i % 2 == 0, update_targets=i % 2 == 0)
                ddpg.update_lr_scheduler()
                writer.add_scalar("train_step_time", timer.end(), global_step.get())

        # if episode.get() % c.ddpg_average_target_int == 0:
        #     ddpg.average_target_parameters()

        if render:
            create_gif_subproc(frames, save_env.get_trial_image_dir() + "/{}_{}".format(episode, global_step))

        local_step.reset()
        episode_finished = False
        logger.info("End episode {} at {}".format(episode, dt.now().strftime("%m/%d-%H:%M:%S")))
