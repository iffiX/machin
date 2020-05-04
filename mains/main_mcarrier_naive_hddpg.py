import time
import torch as t
import torch.nn as nn

from models.frameworks.hddpg import HDDPG
from models.naive.env_walker import Actor, Critic

from utils.logging import default_logger as logger
from utils.image import create_gif
from utils.tensor_board import global_board
from utils.helper_classes import Counter
from utils.prep import prep_dirs_default
from utils.args import get_args

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

agent_num = 3
explore_noise_params = [(0, 0.2)] * action_dim
q_increase_rate = 1
q_decrease_rate = 1
device = t.device("cuda:0")
root_dir = "/data/AI/tmp/multi_agent/walker/hdqn/"
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
    prep_dirs_default(root_dir)
    logger.info("Directories prepared.")
    global_board.init(log_dir + "train_log")
    writer = global_board.writer

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
    # preparations
    env = BipedalMultiCarrier(agent_num=agent_num)

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

            # render configuration
            if episode.get() % profile_int == 0 and global_step.get() > ddpg_warmup_steps:
                render = True
            else:
                render = False
            frames = []

            # model serialization
            if episode.get() % model_save_int == 0:
                ddpg.save(model_dir, save_map, global_step.get())
                logger.info("Saving model parameters, epoch={}, episode={}"
                            .format(epoch, episode))

            # batch size = 1
            episode_begin = time.time()
            actions = t.zeros([1, agent_num * action_dim], device=device)
            total_reward = t.zeros([1, agent_num], device=device)
            state, reward = t.tensor(env.reset(), dtype=t.float32, device=device), 0

            while not episode_finished and local_step.get() <= max_steps:
                global_step.count()
                local_step.count()

                step_begin = time.time()
                with t.no_grad():
                    old_state = state

                    # agent model inference
                    for ag in range(agent_num):
                        if not render:
                            actions[:, ag * action_dim: (ag + 1) * action_dim] = ddpg.act_with_noise(
                                {"state": state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0)},
                                explore_noise_params, mode="normal")
                        else:
                            actions[:, ag * action_dim: (ag + 1) * action_dim] = ddpg.act(
                                {"state": state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0)})

                    actions = t.clamp(actions, min=-1, max=1)
                    state, reward, episode_finished, _ = env.step(actions[0].to("cpu"))

                    if render:
                        frames.append(env.render(mode="rgb_array"))

                    state = t.tensor(state, dtype=t.float32, device=device)
                    reward = t.tensor(reward, dtype=t.float32, device=device).unsqueeze(dim=0)

                    total_reward += reward

                    for ag in range(agent_num):
                        ddpg.store_observe({"state": {"state":
                                                old_state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0).clone()},
                                            "action": {"action": actions[:, ag * action_dim:(ag + 1) * action_dim].clone()},
                                            "next_state": {"state":
                                                state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0).clone()},
                                            "reward": float(reward[0][ag]),
                                            "terminal": episode_finished or local_step.get() == max_steps})

                    writer.add_scalar("action_min", t.min(actions), global_step.get())
                    writer.add_scalar("action_mean", t.mean(actions), global_step.get())
                    writer.add_scalar("action_max", t.max(actions), global_step.get())

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
                    ddpg.update(update_policy=i % 2 == 0, update_targets=i % 2 == 0)
                    ddpg_train_end = time.time()
                    logger.info("DDPG train Step {} completed in {:.3f} s, epoch={}, episode={}".
                                format(i, ddpg_train_end - ddpg_train_begin, epoch, episode, global_step.get()))

            if render:
                create_gif(frames, "{}/log/images/{}_{}_{}".format(root_dir, epoch, episode, global_step.get()))

            local_step.reset()
            episode_finished = False
            episode_end = time.time()
            logger.info("Episode {} completed in {:.3f} s, epoch={}".
                        format(episode, episode_end - episode_begin, epoch))

        episode.reset()