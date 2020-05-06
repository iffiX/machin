import itertools as it
import torch as t
import torch.nn as nn
from datetime import datetime as dt

from models.models.base import NeuralNetworkWrapper as NNW
from models.noise.action_space_noise import add_normal_noise_to_action
from models.frameworks.hddpg import HDDPG
from models.naive.env_walker import Actor, Critic

from utils.logging import default_logger as logger
from utils.image import create_gif
from utils.tensor_board import global_board
from utils.helper_classes import Counter, Timer, Object
from utils.env import Environment
from utils.conf import *
from utils.parallel import Pool, get_context

from env.walker.carrier import BipedalMultiCarrier

# definitions
observe_dim = 28
action_dim = 4

# configs
c = Config()
#c.restart_from_trial = "2020_05_06_21_50_57"
c.max_episodes = 20000
c.max_steps = 2000
c.replay_size = 500000

c.agent_num = 3
c.explore_noise_params = (0, 0.2)
c.q_increase_rate = 1
c.q_decrease_rate = 1
c.device = "cuda:0"
c.root_dir = "/data/AI/tmp/multi_agent/mcarrier/hdqn/"

# train configs
# lr: learning rate, int: interval
# warm up should be less than one epoch
c.ddpg_update_batch_size = 100
c.ddpg_warmup_steps = 200
c.model_save_int = 100  # in episodes
c.profile_int = 50  # in episodes


if __name__ == "__main__":
    args = get_args()
    merge_config(c, args.conf)

    # preparations
    save_env = Environment(c.root_dir, restart_use_trial=c.restart_from_trial)
    if c.restart_from_trial is not None:
        r = c.restart_from_trial
        replace_config(c, load_config_cdict(save_env.get_trial_config_file()))
        save_env.clear_trial_train_log_dir()
        # prevent overwriting
        c.restart_from_trial = r
    else:
        save_config(c, save_env.get_trial_config_file())
    # save_env.remove_trials_older_than(diff_hour=1)
    global_board.init(save_env.get_trial_train_log_dir())
    writer = global_board.writer
    logger.info("Directories prepared.")

    actor = NNW(Actor(observe_dim, action_dim, 1).to(c.device), c.device, c.device)
    actor_t = NNW(Actor(observe_dim, action_dim, 1).to(c.device), c.device, c.device)
    critic = NNW(Critic(observe_dim, action_dim).to(c.device), c.device, c.device)
    critic_t = NNW(Critic(observe_dim, action_dim).to(c.device), c.device, c.device)

    # ctx = get_context("spawn")
    # pool = Pool(processes=2, context=ctx)
    # pool.enable_copy_tensors(True)
    # pool.enable_global(True)
    # actor.share_memory()
    # actor_t.share_memory()
    # critic.share_memory()
    # critic_t.share_memory()
    logger.info("Networks created")

    ddpg = HDDPG(actor, actor_t, critic, critic_t,
                 t.optim.Adam, nn.MSELoss(reduction='sum'),
                 q_increase_rate=c.q_increase_rate,
                 q_decrease_rate=c.q_decrease_rate,
                 discount=0.99,
                 update_rate=0.005,
                 batch_size=c.ddpg_update_batch_size,
                 learning_rate=0.001,
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
        actions = t.zeros([1, c.agent_num * action_dim], device=c.device)
        total_reward = t.zeros([1, c.agent_num], device=c.device)
        state, reward = t.tensor(env.reset(), dtype=t.float32, device=c.device), 0

        while not episode_finished and local_step.get() <= c.max_steps:
            global_step.count()
            local_step.count()

            timer.begin()
            with t.no_grad():
                old_state = state

                # actions.share_memory_()
                # state.share_memory_()

                # never do this:
                # def infer(ag, render, actions, state, ddpg):
                #     if not render:
                #         actions[:, ag * action_dim: (ag + 1) * action_dim] = ddpg.act_with_noise(
                #             {"state": state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0)},
                #             explore_noise_params, mode="normal")
                #     ...
                # because samples will be appended to ddpg, and for class instances, pickle/dill
                # will have to serialize the whole class instance every time, therefore the program
                # will get unbearably slower and slower

                # agent model inference

                # Method 1: multiprocessing pool
                # Performance: 18ms for 10 agents, 500MB memory for each process
                # seems that dill will pickle the whole tensor(actions and state)
                # rather than pickle their reference
                # Major lag: acquire_lock() in queue
                #
                # def infer(ag):
                #     action = actor(state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0))
                #     if not render:
                #         actions[:, ag * action_dim: (ag + 1) * action_dim] = \
                #         add_normal_noise_to_action(action, explore_noise_params)
                #     else:
                #         actions[:, ag * action_dim: (ag + 1) * action_dim] = action
                #
                # pool.map(infer, range(agent_num))

                # Method 2: naive loop
                # Performance: 4ms for 10 agents
                #
                # for ag in range(agent_num):
                #     if not render:
                #         actions[:, ag * action_dim: (ag + 1) * action_dim] = ddpg.act_with_noise(
                #             {"state": state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0)},
                #             explore_noise_params, mode="normal")
                #     else:
                #         actions[:, ag * action_dim: (ag + 1) * action_dim] = ddpg.act(
                #             {"state": state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0)})

                # Method 3: The best way in this simple environment is to vectorize all actors
                # (since they all share the same model)
                # Performance: 0.5ms for 10 agents
                #
                if not render:
                    actions = ddpg.act_with_noise(
                        {"state": state.view(-1, observe_dim)},
                        c.explore_noise_params, mode="normal").view(1, -1)
                else:
                    actions = ddpg.act(
                        {"state": state.view(-1, observe_dim)}).view(1, -1)

                actions = t.clamp(actions, min=-1, max=1)
                state, reward, episode_finished, _ = env.step(actions[0].to("cpu"))

                if render:
                    frames.append(env.render(mode="rgb_array"))

                state = t.tensor(state, dtype=t.float32, device=c.device)
                reward = t.tensor(reward, dtype=t.float32, device=c.device).unsqueeze(dim=0)

                total_reward += reward

                for ag in range(c.agent_num):
                    ddpg.store_observe({
                        "state": {"state": old_state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0).clone()},
                        "action": {"action": actions[:, ag * action_dim:(ag + 1) * action_dim].clone()},
                        "next_state": {"state": state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(0).clone()},
                        "reward": float(reward[0][ag]),
                        "terminal": episode_finished or local_step.get() == c.max_steps
                    })

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
                writer.add_scalar("train_step_time", timer.end(), global_step.get())

        if render:
            create_gif(frames, save_env.get_trial_image_dir() + "/{}_{}".format(episode, global_step))

        local_step.reset()
        episode_finished = False
        logger.info("End episode {} at {}".format(episode, dt.now().strftime("%m/%d-%H:%M:%S")))
