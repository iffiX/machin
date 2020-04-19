import time
import torch as t
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR

from models.frameworks.ddpg import DDPG
from models.noise import OrnsteinUhlenbeckNoise

from utils.logging import default_logger as logger
from utils.image import create_gif
from utils.tensor_board import global_board
from utils.helper_classes import Counter
from utils.prep import prep_dir_default
from utils.args import get_args

from env.walker.single_walker import BipedalWalker

# configs
restart = True
# max_batch = 8
max_epochs = 20
max_episodes = 800
max_steps = 1000
replay_size = 500000
agent_num = 1
noise_range = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
device = t.device("cuda:0")
root_dir = "/data/AI/tmp/multi_agent/walker/naive2/"
model_dir = root_dir + "model/"
log_dir = root_dir + "log/"
save_map = {"actor": "actor",
            "critic": "critic"}

observe_dim = 24
action_dim = 4
# train configs
# lr: learning rate, int: interval
# warm up should be less than one epoch
ddpg_update_int = 1  # in steps
ddpg_update_batch_num = 1
ddpg_warmup_steps = 2000
model_save_int = 100  # in episodes
profile_int = 50  # in episodes


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = t.relu(self.l1(state))
        a = t.relu(self.l2(a))
        a = t.tanh(self.l3(a)) * self.max_action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)

        q = t.relu(self.l1(state_action))
        q = t.relu(self.l2(q))
        q = self.l3(q)
        return q


def gen_learning_rate_func(lr_map):
    def learning_rate_func(step):
        for i in range(len(lr_map) - 1):
            if lr_map[i][0] <= step < lr_map[i + 1][0]:
                return lr_map[i][1]
        return lr_map[-1][1]

    return learning_rate_func


if __name__ == "__main__":
    args = get_args()
    for k, v in args.env.items():
        globals()[k] = v
    total_steps = max_epochs * max_episodes * max_steps

    # preparations
    prep_dir_default(root_dir)
    logger.info("Directories prepared.")
    global_board.init(log_dir + "train_log")
    writer = global_board.writer

    actor = Actor(observe_dim, action_dim, 1).to(device)
    actor_t = Actor(observe_dim, action_dim, 1).to(device)
    critic = Critic(observe_dim, action_dim).to(device)
    critic_t = Critic(observe_dim, action_dim).to(device)

    logger.info("Networks created")

    actor_lr_map = [[0, 1e-3],
                    [total_steps // 3, 1e-3],
                    [total_steps * 2 // 3, 1e-3],
                    [total_steps, 1e-3]]
    critic_lr_map = [[0, 1e-3],
                     [total_steps // 3, 1e-3],
                     [total_steps * 2 // 3, 1e-3],
                     [total_steps, 1e-3]]

    actor_lr_func = gen_learning_rate_func(actor_lr_map)
    critic_lr_func = gen_learning_rate_func(critic_lr_map)

    ddpg = DDPG(actor, actor_t, critic, critic_t,
                t.optim.Adam, nn.MSELoss(reduction='sum'), device,
                discount=0.99,
                batch_num=100,
                lr_scheduler=LambdaLR,
                lr_scheduler_params=[[actor_lr_func], [critic_lr_func]],
                replay_size=replay_size)

    if not restart:
        ddpg.load(root_dir + "/model", save_map)
    logger.info("DDPG framework initialized")

    # training
    # preparations
    env = BipedalWalker()

    # begin training
    # epoch > episode
    epoch = Counter()
    episode = Counter()
    episode_finished = False
    global_step = Counter()
    local_step = Counter()
    noise = OrnsteinUhlenbeckNoise([1], 0.5, 0.1)
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
                ddpg.save(model_dir, save_map, episode.get() + (epoch.get() - 1) * max_episodes)
                logger.info("Saving model parameters, epoch={}, episode={}"
                            .format(epoch, episode))

            # batch size = 1
            episode_begin = time.time()
            actions = t.zeros([1, agent_num * 4], device=device)
            total_reward = t.zeros([1, agent_num], device=device)

            while not episode_finished and local_step.get() <= max_steps:
                global_step.count()
                local_step.count()

                step_begin = time.time()
                state, reward, old_state, old_reward = None, None, None, None
                with t.no_grad():
                    old_state, old_reward = state, reward
                    state, reward, episode_finished, _ = env.step(actions[0].to("cpu"))

                    if render:
                        frames.append(env.render(mode="rgb_array"))

                    state = t.tensor(state, dtype=t.float32, device=device)
                    reward = t.tensor(reward, dtype=t.float32, device=device).unsqueeze(dim=0)

                    total_reward += reward

                    # agent model inference
                    for ag in range(agent_num):
                        actions[:, ag * 4: (ag + 1) * 4] = actor(state[ag * 24: (ag + 1) * 24].unsqueeze(0))

                    if not render:
                        n = float(noise())
                        actions = ddpg.add_noise_to_action(actions,
                                                           noise_range * agent_num,
                                                           n)

                    actions = t.clamp(actions, min=-1, max=1)

                    writer.add_scalar("action_min", t.min(actions), global_step.get())
                    writer.add_scalar("action_mean", t.mean(actions), global_step.get())
                    writer.add_scalar("action_max", t.max(actions), global_step.get())

                    if local_step.get() > 1:
                        for ag in range(agent_num):
                            ddpg.store_observe({"state": {"state": state[ag * 24: (ag + 1) * 24].unsqueeze(0)},
                                                "action": {"action": actions[:, ag * 4:(ag+1)*4]},
                                                "next_state": {"state": state[ag * 24: (ag + 1) * 24].unsqueeze(0)},
                                                "reward": reward[ag],
                                                "terminal": episode_finished or local_step.get() == max_steps})

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
                for i in range(ddpg_update_batch_num):
                    ddpg_train_begin = time.time()
                    ddpg.update()
                    ddpg.update_lr_scheduler()
                    ddpg_train_end = time.time()
                    logger.info("DDPG train Step {} completed in {:.3f} s, epoch={}, episode={}".
                                format(i, ddpg_train_end - ddpg_train_begin, epoch, episode))

            if render:
                create_gif(frames, "{}/log/images/{}_{}".format(root_dir, epoch, episode))

            local_step.reset()
            episode_finished = False
            episode_end = time.time()
            logger.info("Episode {} completed in {:.3f} s, epoch={}".
                        format(episode, episode_end - episode_begin, epoch))

        episode.reset()
