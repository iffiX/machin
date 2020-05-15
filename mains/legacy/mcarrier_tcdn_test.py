import time
import torch as t
import torch.nn as nn

from models.frames.ddpg import DDPG
from models.tcdn.actor import SwarmActor, WrappedActorNet
from models.tcdn.critic import SwarmCritic, WrappedCriticNet
from models.tcdn.negotiatior import SwarmNegotiator
from models.tcdn.agent import SwarmAgent

from utils.logging import default_logger as logger
from utils.image import create_gif
from utils.helper_classes import Counter
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
replay_size = 400000

agent_num = 1
history_depth = 1
neighbors = [-1, 1]
neighbor_num = len(neighbors)
nego_mean_anneal = 0.3
nego_theta_anneal = 0.1
nego_rounds = 0
device = t.device("cuda:0")
load_dir = "/data/AI/tmp/multi_agent/walker/tcdn/model/"
save_map = {}


if __name__ == "__main__":
    args = get_args()
    for k, v in args.env.items():
        globals()[k] = v
    total_steps = max_epochs * max_episodes * max_steps

    base_actor = SwarmActor(observe_dim, action_dim, history_depth, neighbor_num, True, device)
    base_actor_t = SwarmActor(observe_dim, action_dim, history_depth, neighbor_num, True, device)
    negotiator = SwarmNegotiator(observe_dim, action_dim, history_depth, neighbor_num, True, device)
    negotiator_t = SwarmNegotiator(observe_dim, action_dim, history_depth, neighbor_num, True, device)

    # currently, all agents share the same two networks
    # TODO: implement K-Sub policies
    actor = WrappedActorNet(base_actor, negotiator)
    actor_t = WrappedActorNet(base_actor_t, negotiator_t)
    critic = WrappedCriticNet(SwarmCritic(observe_dim, action_dim, history_depth, neighbor_num, device))
    critic_t = WrappedCriticNet(SwarmCritic(observe_dim, action_dim, history_depth, neighbor_num, device))

    logger.info("Networks created")

    # only used to load model
    ddpg = DDPG(actor, actor_t, critic, critic_t,
                t.optim.Adam, nn.MSELoss(reduction='sum'), device)

    ddpg.load(load_dir, save_map)
    logger.info("DDPG framework initialized")

    # evaluation
    # preparations
    env = BipedalMultiCarrier(agent_num)
    agents = [SwarmAgent(base_actor, negotiator, len(neighbors), action_dim, observe_dim, history_depth,
                         mean_anneal=nego_mean_anneal, theta_anneal=nego_theta_anneal,
                         batch_size=1, contiguous=True, device=device)
              for i in range(agent_num)]

    # begin evaluation
    # epoch > episode
    episode_finished = False
    local_step = Counter()

    #check_model(writer, critic, global_step, name="critic")
    #check_model(writer, base_actor, global_step, name="actor")

    logger.info("Begin testing")

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

    # render configuration
    frames = []

    # batch size = 1
    episode_begin = time.time()
    actions = t.zeros([1, agent_num * action_dim], device=device)
    total_reward = t.zeros([1, agent_num], device=device)
    state, reward = t.tensor(env.reset(), dtype=t.float32, device=device), 0

    while not episode_finished and local_step.get() <= max_steps:
        local_step.count()

        step_begin = time.time()
        with t.no_grad():

            # agent model inference
            for ag in range(agent_num):
                agents[ag].set_observe(state[ag * observe_dim: (ag + 1) * observe_dim].unsqueeze(dim=0))

            for ag in range(agent_num):
                print("agent {} act:".format(ag))
                agents[ag].act_step(local_step.get())

            for i in range(nego_rounds):
                for ag in range(agent_num):
                    print("agent {} nego {}:".format(ag, i))
                    agents[ag].negotiate_step()

            for ag in range(agent_num):
                actions[:, ag * action_dim: (ag + 1) * action_dim] = agents[ag].final_step()

            actions = t.clamp(actions, min=-1, max=1)
            state, reward, episode_finished, info = env.step(actions[0].to("cpu"))

            frames.append(env.render(mode="rgb_array"))

            state = t.tensor(state, dtype=t.float32, device=device)
            reward = t.tensor(reward, dtype=t.float32, device=device).unsqueeze(dim=0)

            total_reward += reward

            for agent, r in zip(agents, reward[0]):
                agent.set_reward(r.view(1, 1))

            old_samples = [agent.get_sample() for agent in agents]

        for agent in agents:
            agent.update_history(local_step.get())
            agent.reset_negotiate()
        step_end = time.time()
        logger.info("Step {} completed in {:.3f} s".format(local_step, step_end - step_begin))

    create_gif(frames, "{}/test".format(load_dir))
    episode_end = time.time()
    logger.info("Episode completed in {:.3f} s".format(episode_end - episode_begin))
