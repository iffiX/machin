import torch as t
import torch.nn as nn

from datetime import datetime as dt

from models.models.base import StaticModuleWrapper as MW
from models.frameworks.ppo import PPO
from models.naive.env_walker_ppo import Actor, Critic

from utils.logging import default_logger as logger
from utils.image import create_gif
from utils.tensor_board import global_board
from utils.helper_classes import Counter, Timer
from utils.conf import Config
from utils.env import Environment
from utils.prep import prep_args

from env.walker.single_walker import BipedalWalker

# definitions
observe_dim = 24
action_dim = 4

# configs
c = Config()
#c.restart_from_trial = "2020_05_09_15_00_31"
c.max_episodes = 5000
c.max_steps = 1000
c.replay_size = 10000

c.explore_noise_params = (0, 0.2)
c.device = "cuda:0"
c.root_dir = "/data/AI/tmp/multi_agent/walker/naive_ppo/"

# train configs
# lr: learning rate, int: interval
c.ppo_update_batch_size = 100
c.ppo_update_times = 50
c.ppo_update_int = 5  # = the number of episodes are stored in ppo replay buffer
c.model_save_int = 100  # in episodes
c.profile_int = 50  # in episodes

if __name__ == "__main__":
    save_env = Environment(c.root_dir, restart_use_trial=c.restart_from_trial)
    prep_args(c, save_env)

    # save_env.remove_trials_older_than(diff_hour=1)
    global_board.init(save_env.get_trial_train_log_dir())
    writer = global_board.writer
    logger.info("Directories prepared.")

    actor = MW(Actor(observe_dim, action_dim, 1).to(c.device), c.device, c.device)
    critic = MW(Critic(observe_dim).to(c.device), c.device, c.device)
    logger.info("Networks created")

    # default replay buffer storage is main cpu mem
    # when stored in main mem, takes about 0.65e-3 sec to move result from gpu to cpu,
    ppo = PPO(actor, critic,
              t.optim.Adam, nn.MSELoss(reduction='sum'),
              replay_device=c.device,
              replay_size=c.replay_size,
              discount=0.99,
              update_times=c.ppo_update_times,
              batch_size=c.ppo_update_batch_size,
              learning_rate=0.001)

    if c.restart_from_trial is not None:
        ppo.load(save_env.get_trial_model_dir())
    logger.info("PPO framework initialized")

    # training
    # preparations
    env = BipedalWalker()

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
        if episode.get() % c.profile_int == 0:
            render = True
        else:
            render = False
        frames = []

        # model serialization
        if episode.get() % c.model_save_int == 0:
            ppo.save(save_env.get_trial_model_dir(), version=episode.get())
            logger.info("Saving model parameters, episode={}".format(episode))

        # batch size = 1
        total_reward = 0
        state, reward = t.tensor(env.reset(), dtype=t.float32, device=c.device), 0

        while not episode_finished and local_step.get() <= c.max_steps:
            global_step.count()
            local_step.count()

            timer.begin()
            with t.no_grad():
                old_state = state

                # agent model inference
                action, prob = ppo.act({"state": state.unsqueeze(0)})

                state, reward, episode_finished, _ = env.step(action[0].to("cpu"))

                if render:
                    frames.append(env.render(mode="rgb_array"))

                state = t.tensor(state, dtype=t.float32, device=c.device)

                total_reward += reward

                ppo.store_observe({"state": {"state": old_state.unsqueeze(0).clone()},
                                   "action": {"action": action.clone()},
                                   "next_state": {"state": state.unsqueeze(0).clone()},
                                   "reward": float(reward),
                                   "terminal": episode_finished or local_step.get() == c.max_steps,
                                   "action_log_prob": float(prob)
                                   })

                writer.add_scalar("action_min", t.min(action), global_step.get())
                writer.add_scalar("action_mean", t.mean(action), global_step.get())
                writer.add_scalar("action_max", t.max(action), global_step.get())

            writer.add_scalar("step_time", timer.end(), global_step.get())
            writer.add_scalar("episodic_reward", reward, global_step.get())
            writer.add_scalar("episodic_sum_reward", total_reward, global_step.get())
            writer.add_scalar("episode_length", local_step.get(), global_step.get())

        logger.info("Sum reward: {}, episode={}".format(total_reward, episode))

        if episode.get() % c.ppo_update_int == 0:
            timer.begin()
            ppo.update()
            ppo.update_lr_scheduler()
            writer.add_scalar("train_step_time", timer.end(), episode.get())
            logger.info("Train end, time = {:.2f} s, episode={}".format(timer.end(), episode))

        if render:
            create_gif(frames, save_env.get_trial_image_dir() + "/{}_{}".format(episode, global_step))

        local_step.reset()
        episode_finished = False
        logger.info("End episode {} at {}".format(episode, dt.now().strftime("%m/%d-%H:%M:%S")))
