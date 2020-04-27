import numpy as np
import env.magent as magent


def generate_map(env, map_size, agent_ratio, left_agents_handle, right_agents_handle):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * agent_ratio
    gap = 3

    # Note: in position (x, y, 0) array, the last dimension is agent's initial direction
    # left
    n = init_num
    side = int(np.sqrt(n)) * 2
    pos = []
    for x in range(width // 2 - gap - side, width // 2 - gap - side + side, 2):
        for y in range((height - side) // 2, (height - side) // 2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(left_agents_handle, method="custom", pos=pos)

    # right
    n = init_num
    side = int(np.sqrt(n)) * 2
    pos = []
    for x in range(width // 2 + gap, width // 2 + gap + side, 2):
        for y in range((height - side) // 2, (height - side) // 2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(right_agents_handle, method="custom", pos=pos)


def generate_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    small_agent = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1,

         'step_reward': -0.005, 'kill_reward': 5, 'dead_penalty': -0.1, 'attack_penalty': -0.1,
         })

    g0 = cfg.add_group(small_agent)
    g1 = cfg.add_group(small_agent)

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    # reward shaping to encourage attack
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=0.2)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=0.2)

    return cfg


def build_comm_network(id, pos, neighbor_num, agents):
    dist = np.sqrt(np.sum((pos - np.expand_dims(pos, axis=1))**2, axis=2))
    nearest = id[np.argsort(dist, axis=1)][:, 1:]  # exclude self, i.e. first column
    # currently, no communication range limit
    peer_num = min(nearest.shape[1], neighbor_num)
    for i in id:
        agents[i].set_neighbors([agents[nearest[i, n]] for n in range(peer_num)])