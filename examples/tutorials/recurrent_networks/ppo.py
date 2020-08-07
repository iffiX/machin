from machin.env.utils.openai_gym import disable_view_window
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
from torch.distributions import Categorical

import gym
import torch as t
import torch.nn as nn

from util import convert
from history import History

# configurations
env = gym.make("Frostbite-ram-v0")
action_num = env.action_space.n
max_episodes = 20000
history_depth = 4

# disable view window in rendering
disable_view_window()


class Actor(nn.Module):
    def __init__(self, history_depth, action_num):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(128 * history_depth, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_num)

    def forward(self, mem, action=None):
        a = t.relu(self.fc1(mem.flatten(start_dim=1)))
        a = t.relu(self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = (action
               if action is not None
               else dist.sample())
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, history_depth):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(128 * history_depth, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, mem):
        v = t.relu(self.fc1(mem.flatten(start_dim=1)))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


if __name__ == "__main__":
    actor = Actor(history_depth, action_num).to("cuda:0")
    critic = Critic(history_depth).to("cuda:0")

    ppo = PPO(actor, critic,
              t.optim.Adam,
              nn.MSELoss(reduction='sum'),
              actor_learning_rate=1e-5,
              critic_learning_rate=1e-4)

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = convert(env.reset())
        history = History(history_depth, (1, 128))

        tmp_observations = []
        while not terminal:
            step += 1
            with t.no_grad():
                history.append(state)
                # agent model inference
                action = ppo.act({"mem": history.get()})[0]
                state, reward, terminal, _ = env.step(action.item())
                state = convert(state)
                total_reward += reward

                old_history = history.get()
                new_history = history.append(state).get()
                tmp_observations.append({
                    "state": {"mem": old_history},
                    "action": {"action": action},
                    "next_state": {"mem": new_history},
                    "reward": reward,
                    "terminal": terminal
                })

        # update
        ppo.store_episode(tmp_observations)
        ppo.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)

        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))
