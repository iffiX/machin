import time
import torch as t
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from models.frameworks.maddpg import MADDPG
from models.naive.env_walker import Actor

from utils.logging import default_logger as logger
from utils.image import create_gif
from utils.tensor_board import global_board
from utils.helper_classes import Counter
from utils.prep import prep_dir_default
from utils.args import get_args
from utils.train import gen_learning_rate_func

from env.walker.carrier import BipedalMultiCarrier

# definitions
observe_dim = 28
action_dim = 4

# configs
restart = True
clear_old = True
max_epochs = 20
max_episodes = 1000
max_steps = 1000
replay_size = 500000

agent_num = 2
explore_noise_params = [(0, 0.2)] * action_dim
device = t.device("cuda:0")
root_dir = "/data/AI/tmp/multi_agent/walker/maddpg/"
model_dir = root_dir + "model/"
log_dir = root_dir + "log/"
save_map = {}

# train configs
# lr: learning rate, int: interval
# warm up should be less than one epoch
ddpg_update_batch_size = 256
ddpg_warmup_steps = 2000
ddpg_average_target_int = 100
model_save_int = 50  # in episodes
profile_int = 10  # in episodes


class Critic(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim):
        super(Critic, self).__init__()
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        st_dim = state_dim * agent_num
        act_dim = action_dim * agent_num

        self.fc1 = nn.Linear(st_dim, 1024)
        self.fc2 = nn.Linear(1024 + act_dim, 512)
        self.fc3 = nn.Linear(512, 300)
        self.fc4 = nn.Linear(300, 1)

    # obs: batch_size * obs_dim
    def forward(self, all_states, all_actions):
        q = t.relu(self.fc1(all_states))
        q = t.cat([q, all_actions], dim=1)
        q = t.relu(self.fc2(q))
        q = t.relu(self.fc3(q))
        q = self.fc4(q)
        return q
    

if __name__ == "__main__":
    args = get_args()
    for k, v in args.env.items():
        globals()[k] = v
    total_steps = max_epochs * max_episodes * max_steps

    # preparations
    prep_dir_default(root_dir, clear_old=clear_old)
    logger.info("Directories prepared.")
    global_board.init(log_dir + "train_log")
    writer = global_board.writer

    actors = [Actor(observe_dim, action_dim, 1).to(device) for i in range(agent_num)]
    actor_ts = [Actor(observe_dim, action_dim, 1).to(device) for i in range(agent_num)]
    critics = [Critic(agent_num, observe_dim, action_dim).to(device) for i in range(agent_num)]
    critic_ts = [Critic(agent_num, observe_dim, action_dim).to(device) for i in range(agent_num)]

    logger.info("Networks created")

    actor_lr_func = gen_learning_rate_func([[0, 5e-4]])
    critic_lr_func = gen_learning_rate_func([[0, 1e-3]])
    ddpg = MADDPG(
                actors, actor_ts, critics, critic_ts,
                t.optim.Adam, nn.MSELoss(reduction='sum'), device,
                discount=0.99,
                update_rate=1e-3,
                batch_size=ddpg_update_batch_size,
                lr_scheduler=LambdaLR,
                lr_scheduler_params=[[actor_lr_func], [critic_lr_func]],
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
                                ag,
                                {"state": state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0)},
                                explore_noise_params, mode="normal")
                        else:
                            actions[:, ag * action_dim: (ag + 1) * action_dim] = ddpg.act(
                                ag,
                                {"state": state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0)})

                    actions = t.clamp(actions, min=-1, max=1)
                    state, reward, episode_finished, _ = env.step(actions[0].to("cpu"))

                    if render:
                        frames.append(env.render(mode="rgb_array"))

                    state = t.tensor(state, dtype=t.float32, device=device)
                    reward = t.tensor(reward, dtype=t.float32, device=device).unsqueeze(dim=0)

                    total_reward += reward

                    for ag in range(agent_num):
                        ddpg.store_observe({"state": {"state": old_state[ag * observe_dim: (ag + 1) * observe_dim]
                                                               .unsqueeze(0).clone(),
                                                      "all_states": old_state.unsqueeze(0).clone()},
                                            "action": {"all_actions": actions.clone()},
                                            "next_state": {"state": state[ag * observe_dim: (ag + 1) * observe_dim]
                                                                    .unsqueeze(0).clone(),
                                                           "all_states": state.unsqueeze(0).clone()},
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
                    ddpg.update(update_policy=i % 2 == 0)
                    ddpg.update_lr_scheduler()
                    ddpg_train_end = time.time()
                    logger.info("DDPG train Step {} completed in {:.3f} s, epoch={}, episode={}".
                                format(i, ddpg_train_end - ddpg_train_begin, epoch, episode))

            # if episode.get() % ddpg_average_target_int == 0:
            #     ddpg.average_target_parameters()

            if render:
                create_gif(frames, "{}/log/images/{}_{}_{}".format(root_dir, epoch, episode, global_step.get()))

            local_step.reset()
            episode_finished = False
            episode_end = time.time()
            logger.info("Episode {} completed in {:.3f} s, epoch={}".
                        format(episode, episode_end - episode_begin, epoch))

        episode.reset()
