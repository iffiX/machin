import importlib as imp
import random as r
import torch as t
from datetime import datetime as dt

from utils.logging import default_logger as logger
from utils.tensor_board import global_board
from utils.save_env import SaveEnv
from utils.prep import prep_args
from utils.helper_classes import Counter

from env.magent_helper import *

from .utils import draw_agent_num_figure

# configs
max_episodes = 100
max_steps = 500
map_size = 50
agent_ratio = 0.04
load_framework1 = "naive_hddpg"
load_framework2 = "naive_ppo_parallel"
load_trial1 = "2020_05_06_21_50_57"
load_trial2 = "2020_05_06_21_50_57"
test_root_dir = ""


def load_framework(name):
    module = imp.import_module(".magent_" + name)
    return module.c, module.create_models, module.run_agents


if __name__ == "__main__":
    c1, create_models1, run_agents1 = load_framework(load_framework1)
    c2, create_models2, run_agents2 = load_framework(load_framework1)
    save_env1 = SaveEnv(c1.root_dir, restart_use_trial=load_trial1)
    prep_args(c1, save_env1)
    save_env2 = SaveEnv(c2.root_dir, restart_use_trial=load_trial2)
    prep_args(c2, save_env2)

    c1.restart_from_trial = load_trial1
    framework1 = create_models1()
    logger.info("Framework 1 initialized")

    c2.restart_from_trial = load_trial2
    framework2 = create_models2()
    logger.info("Framework 2 initialized")

    operators = [(framework1, run_agents1, load_framework1),
                 (framework2, run_agents2, load_framework2)]

    # testing
    # preparations
    config = generate_combat_config(map_size)
    env = magent.GridWorld(config, map_size=map_size)
    env.reset()

    global_board.init(test_root_dir)
    writer = global_board.writer
    logger.info("Directories prepared.")

    # begin training
    episode = Counter()
    episode_finished = False
    wins = [0, 0]

    while episode < max_episodes:
        episode.count()
        logger.info("Begin episode {} at {}".format(episode, dt.now().strftime("%m/%d-%H:%M:%S")))

        # environment initialization
        env.reset()
        env.set_render_dir(test_root_dir)

        group_handles = env.get_handles()
        generate_combat_map(env, map_size, agent_ratio, group_handles[0], group_handles[1])

        # batch size = 1
        total_reward = [0, 0]
        agent_real_nums = [[], []]
        r.shuffle(operators)

        local_step = Counter()
        episode_finished = False

        while not episode_finished and local_step.get() <= max_steps:
            local_step.count()

            with t.no_grad():
                for g in (0, 1):
                    agent_real_num, *_ = operators[g][1](env, operators[g][0], group_handles[g])
                    agent_real_nums[g].append(agent_real_num)

                episode_finished = env.step()
                # reward and must be got before clear_dead() !
                reward = [env.get_reward(h) for h in group_handles]

                total_reward[0] += np.mean(reward[0])
                total_reward[1] += np.mean(reward[1])

                env.render()
                env.clear_dead()

        if operators[0][2] == load_framework2:
            total_reward[0], total_reward[1] = total_reward[1], total_reward[0]
            operators[0], operators[1] = operators[1], operators[0]
            agent_real_nums[0], agent_real_nums[1] = agent_real_nums[1], agent_real_nums[0]

        if total_reward[0] < total_reward[1]:
            wins[1] += 1
        elif total_reward[0] > total_reward[1]:
            wins[0] += 1

        logger.info("{} Sum reward: {}, episode={}".format(operators[0][2], total_reward[0], episode))
        logger.info("{} Sum reward: {}, episode={}".format(operators[1][2], total_reward[1], episode))
        writer.add_scalar("{}_sum_reward".format(operators[0][2]), total_reward[0], episode.get())
        writer.add_scalar("{}_sum_reward".format(operators[1][2]), total_reward[1], episode.get())
        writer.add_figure("agent_num", draw_agent_num_figure(agent_real_nums), episode.get())
        writer.add_scalar("episode_length", local_step, episode.get())

        local_step.reset()
        episode_finished = False
        logger.info("End episode {} at {}".format(episode, dt.now().strftime("%m/%d-%H:%M:%S")))
    logger.info("{} wins {:.2f}%, {} wins {:.2f}%, draw {:.2f}".format(
        load_framework1, wins[0] * 100 / max_episodes,
        load_framework2, wins[1] * 100 / max_episodes,
        (1 - sum(wins)) * 100 / max_episodes
    ))
