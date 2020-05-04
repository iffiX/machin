import torch as t
import torch.nn as nn

from datetime import datetime as dt

from models.models.base import NeuralNetworkWrapper as NNW
from models.frameworks.ddpg_td3 import DDPG_TD3
from models.naive.env_walker import Actor, Critic

from utils.logging import default_logger as logger
from utils.image import create_gif
from utils.tensor_board import global_board
from utils.helper_classes import Counter, Timer
from utils.env import Environment
from utils.args import get_args

from env.walker.single_walker import BipedalWalker

# definitions
observe_dim = 24
action_dim = 4

# configs
restart = True
max_episodes = 5000
max_steps = 1000
replay_size = 500000
# or: explore_noise_params = (0, 0.2)
explore_noise_params = [(0, 0.2)] * action_dim
policy_noise_params = (0, 0.2)
device = t.device("cuda:0")
root_dir = "/data/AI/tmp/multi_agent/walker/naive_ddpg_td3/"

# train configs
# lr: learning rate, int: interval
# warm up should be less than one epoch
ddpg_update_batch_size = 100
ddpg_warmup_steps = 200
model_save_int = 100  # in episodes
profile_int = 50  # in episodes


def policy_noise_func(actions, *args):
    global explore_noise_params
    noise = t.zeros_like(actions)
    noise = noise.data.normal_(*policy_noise_params)
    noise = t.clamp(noise, -0.5, 0.5)
    actions = actions + noise
    actions = t.clamp(actions, min=-1, max=1)
    return actions


if __name__ == "__main__":
    args = get_args()
    for k, v in args.env.items():
        globals()[k] = v

    # preparations
    save_env = Environment(root_dir)
    #save_env.remove_trials_older_than(diff_hour=1)
    global_board.init(save_env.get_trial_train_log_dir())
    writer = global_board.writer
    logger.info("Directories prepared.")

    # An example where each actor and critic is placed on a difeerent device
    # actor = NNW(Actor(observe_dim, action_dim, 1), "cpu", "cpu")
    # actor_t = NNW(Actor(observe_dim, action_dim, 1).to(device), device, device)
    # critic = NNW(Critic(observe_dim, action_dim).to(device), device, device)
    # critic_t = NNW(Critic(observe_dim, action_dim), "cpu", "cpu")

    actor = NNW(Actor(observe_dim, action_dim, 1).to(device), device, device)
    actor_t = NNW(Actor(observe_dim, action_dim, 1).to(device), device, device)
    critic = NNW(Critic(observe_dim, action_dim).to(device), device, device)
    critic_t = NNW(Critic(observe_dim, action_dim).to(device), device, device)
    critic2 = NNW(Critic(observe_dim, action_dim).to(device), device, device)
    critic2_t = NNW(Critic(observe_dim, action_dim).to(device), device, device)

    logger.info("Networks created")

    # default replay buffer storage is main cpu mem
    ddpg = DDPG_TD3(actor, actor_t, critic, critic_t, critic2, critic2_t,
                    t.optim.Adam, nn.MSELoss(reduction='sum'),
                    discount=0.99,
                    update_rate=0.005,
                    batch_size=ddpg_update_batch_size,
                    learning_rate=0.001,
                    policy_noise_func=policy_noise_func)

    if not restart:
        ddpg.load(save_env.get_trial_model_dir())
    logger.info("DDPG framework initialized")

    # training
    # preparations
    env = BipedalWalker()

    # begin training
    # epoch > episode
    episode = Counter()
    episode_finished = False
    global_step = Counter()
    local_step = Counter()
    timer = Timer()
    while episode < max_episodes:
        episode.count()
        logger.info("Begin episode {} at {}".format(episode, dt.now().strftime("%m/%d-%H:%M:%S")))

        # environment initialization
        env.reset()

        # render configuration
        if episode.get() % profile_int == 0 and global_step.get() > ddpg_warmup_steps:
            render = True
        else:
            render = False
        frames = []

        # model serialization
        if episode.get() % model_save_int == 0:
            ddpg.save(save_env.get_trial_model_dir(), version=episode.get())
            logger.info("Saving model parameters, episode={}".format(episode))

        # batch size = 1
        total_reward = 0
        state, reward = t.tensor(env.reset(), dtype=t.float32, device=device), 0

        while not episode_finished and local_step.get() <= max_steps:
            global_step.count()
            local_step.count()

            timer.begin()
            with t.no_grad():
                old_state = state

                # agent model inference
                if not render:
                    actions = ddpg.act_with_noise({"state": state.unsqueeze(0)},
                                                  explore_noise_params, mode="normal")
                else:
                    actions = ddpg.act({"state": state.unsqueeze(0)})

                actions = t.clamp(actions, min=-1, max=1)
                state, reward, episode_finished, _ = env.step(actions[0].to("cpu"))

                if render:
                    frames.append(env.render(mode="rgb_array"))

                state = t.tensor(state, dtype=t.float32, device=device)

                total_reward += reward

                ddpg.store_observe({"state": {"state": old_state.unsqueeze(0).clone()},
                                    "action": {"action": actions.clone()},
                                    "next_state": {"state": state.unsqueeze(0).clone()},
                                    "reward": float(reward),
                                    "terminal": episode_finished or local_step.get() == max_steps})

                writer.add_scalar("action_min", t.min(actions), global_step.get())
                writer.add_scalar("action_mean", t.mean(actions), global_step.get())
                writer.add_scalar("action_max", t.max(actions), global_step.get())

            writer.add_scalar("step_time", timer.end(), global_step.get())
            writer.add_scalar("episodic_reward", reward, global_step.get())
            writer.add_scalar("episodic_sum_reward", total_reward, global_step.get())
            writer.add_scalar("episode_length", local_step.get(), global_step.get())

        logger.info("Sum reward: {}, episode={}".format(total_reward, episode))

        if global_step.get() > ddpg_warmup_steps:
            for i in range(local_step.get()):
                timer.begin()
                ddpg.update(update_policy=i % 2 == 0, update_targets=i % 2 == 0)
                writer.add_scalar("train_step_time", timer.end(), global_step.get())

        if render:
            create_gif(frames, save_env.get_trial_image_dir() + "/{}_{}".format(episode, global_step))

        local_step.reset()
        episode_finished = False
        logger.info("End episode {} at {}".format(episode, dt.now().strftime("%m/%d-%H:%M:%S")))
