import time
import torch as t
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR

from models.frameworks.ddpg import DDPG
from models.tcdn.actor import SwarmActor, WrappedActorNet
from models.tcdn.critic import SwarmCritic, WrappedCriticNet
from models.tcdn.negotiatior import SwarmNegotiator
from models.tcdn.agent import SwarmAgent

from utils.logging import default_logger as logger
from utils.tensor_board import global_board
from utils.helper_classes import Counter
from utils.prep import prep_dir_default
from utils.args import get_args

from env.walker.multi_walker import BipedalMultiWalker

# max_batch = 8
max_epochs = 20
max_episodes = 800
max_steps = 1000
theta = 1
replay_size = 1000000
history_depth = 10
agent_num = 1
neighbors = [-1, 1]
noise_range = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
mean_anneal = 0.3
theta_anneal = 0.1
device = t.device("cuda:0")
root_dir = "/data/AI/tmp/multi_agent/walker/tcdn/"
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
profile_int = 10  # in episodes


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

    base_actor = SwarmActor(observe_dim, action_dim, history_depth, len(neighbors), True, device)
    base_actor_t = SwarmActor(observe_dim, action_dim, history_depth, len(neighbors), True, device)
    negotiator = SwarmNegotiator(observe_dim, action_dim, history_depth, len(neighbors), True, device)
    negotiator_t = SwarmNegotiator(observe_dim, action_dim, history_depth, len(neighbors), True, device)

    # currently, all agents share the same two networks
    # TODO: implement K-Sub policies
    actor = WrappedActorNet(base_actor, negotiator)
    actor_t = WrappedActorNet(base_actor_t, negotiator_t)
    critic = WrappedCriticNet(SwarmCritic(observe_dim, action_dim, history_depth, len(neighbors), device))
    critic_t = WrappedCriticNet(SwarmCritic(observe_dim, action_dim, history_depth, len(neighbors), device))

    logger.info("Networks created")

    actor_lr_map = [[0, 1e-4],
                    [total_steps // 3, 1e-4],
                    [total_steps * 2 // 3, 1e-4],
                    [total_steps, 1e-4]]
    critic_lr_map = [[0, 1e-4],
                     [total_steps // 3, 1e-4],
                     [total_steps * 2 // 3, 1e-4],
                     [total_steps, 1e-4]]

    actor_lr_func = gen_learning_rate_func(actor_lr_map)
    critic_lr_func = gen_learning_rate_func(critic_lr_map)

    ddpg = DDPG(actor, actor_t, critic, critic_t,
                t.optim.Adam, nn.MSELoss(reduction='sum'), device,
                lr_scheduler=LambdaLR,
                lr_scheduler_params=[[actor_lr_func], [critic_lr_func]],
                replay_size=replay_size,
                batch_num=1)

    ddpg.load(root_dir + "/model", save_map)
    logger.info("DDPG framework initialized")

    # training
    # preparations
    env = BipedalMultiWalker(agent_num)
    agents = [SwarmAgent(base_actor, negotiator, len(neighbors), action_dim, observe_dim,
                         history_depth, mean_anneal, theta_anneal, 1, True, device)
              for i in range(agent_num)]

    # begin training
    # epoch > episode
    episode_finished = False

    env.reset()
    for agent in agents:
        agent.reset()

    ### currently, agents have fixed communication topology
    for i in range(agent_num):
        agent_neighbors = []
        for j in neighbors:
            index = i + j
            if agent_num > index >= 0:
                agent_neighbors.append(index)
        agents[i].set_neighbors([agents[n] for n in agent_neighbors])

    # batch size = 1
    episode_begin = time.time()
    actions = t.zeros([1, agent_num * 4], device=device)
    total_reward = t.zeros([1, agent_num], device=device)
    local_step = Counter()

    while not episode_finished and local_step.get() <= max_steps:
        local_step.count()

        step_begin = time.time()
        with t.no_grad():
            old_samples = [agent.get_sample() for agent in agents]

            state, reward, episode_finished, info = env.step(actions[0].to("cpu"))

            state = t.tensor(state, dtype=t.float32, device=device)
            reward = t.tensor(reward, dtype=t.float32, device=device).unsqueeze(dim=0)

            total_reward += reward

            # agent model inference
            for ag in range(agent_num):
                agents[ag].set_observe(state[ag * 24: (ag + 1) * 24].unsqueeze(dim=0))

            for ag in range(agent_num):
                agents[ag].act_step()

            #while all([agent.get_negotiation_rate() > 0.01 for agent in agents]):
            #    for ag in range(agent_num):
            #        agents[ag].negotiate_step()

            for ag in range(agent_num):
                actions[:, ag * 4: (ag + 1) * 4] = agents[ag].final_step()

            print("actions:")
            print(actions)

            if local_step.get() > 1:
                for agent, r in zip(agents, reward[0]):
                    agent.set_reward(r)

        for agent in agents:
            if local_step.get() > 1:
                agent.update_history()
            agent.reset_negotiate()
