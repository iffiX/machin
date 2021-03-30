from machin.env.utils.openai_gym import disable_view_window
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
from torch.distributions import Categorical

import gym
import torch as t
import torch.nn as nn

from util import convert

# configurations
env = gym.make("Frostbite-ram-v0")
action_num = env.action_space.n
max_episodes = 20000

# disable view window in rendering
disable_view_window()


class RecurrentActor(nn.Module):
    def __init__(self, action_num):
        super().__init__()
        self.gru = nn.GRU(128, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, action_num)

    def forward(self, mem, hidden, action=None):
        hidden = hidden.transpose(0, 1)
        a, hidden = self.gru(mem.unsqueeze(1), hidden)
        a = self.fc2(t.relu(self.fc1(t.relu(a.flatten(start_dim=1)))))
        probs = t.softmax(a, dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy, hidden


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, mem):
        v = t.relu(self.fc1(mem))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


if __name__ == "__main__":
    actor = RecurrentActor(action_num).to("cuda:0")
    critic = Critic().to("cuda:0")

    rppo = PPO(
        actor,
        critic,
        t.optim.Adam,
        nn.MSELoss(reduction="sum"),
        actor_learning_rate=1e-5,
        critic_learning_rate=1e-4,
    )

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        hidden = t.zeros([1, 1, 256])
        state = convert(env.reset())

        tmp_observations = []
        while not terminal:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                old_hidden = hidden
                action, _, _, hidden = rppo.act({"mem": state, "hidden": hidden})
                state, reward, terminal, _ = env.step(action.item())
                state = convert(state)
                total_reward += reward

                tmp_observations.append(
                    {
                        "state": {"mem": old_state, "hidden": old_hidden},
                        "action": {"action": action},
                        "next_state": {"mem": state, "hidden": hidden},
                        "reward": reward,
                        "terminal": terminal,
                    }
                )

        # update
        rppo.store_episode(tmp_observations)
        rppo.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1

        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")
