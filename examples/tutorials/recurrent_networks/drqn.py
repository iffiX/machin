from machin.env.utils.openai_gym import disable_view_window
from machin.frame.algorithms import DQN
from machin.utils.logging import default_logger as logger
from skimage.transform import resize
from skimage.color import rgb2gray

import numpy as np
import torch as t
import torch.nn as nn
import gym

# configurations
env = gym.make("Frostbite-v0")
max_episodes = 20000
history_depth = 4

# disable view window in rendering
disable_view_window()


def resize_and_convert(img: np.ndarray):
    # image is H, W, 3
    return (resize(rgb2gray(img), (84, 84))
            .reshape(1, 84, 84)
            .astype(np.float32)
            / 255)


# model definition
class RecurrentQNet(nn.Module):
    def __init__(self, history_depth):
        super(RecurrentQNet, self).__init__()
        self.conv1 = nn.Conv2d(history_depth, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # input screen size is 1x(history_depth x 1)x84x84 (single channel)
        # therefore output of conv3 should be 1x64x7x7
        self.fc1 = nn.Linear(64*7*7, 512)

        # action num is 18 for atari games
        self.fc2 = nn.Linear(512, 18)

    def forward(self, screen):
        a = self.conv3(self.conv2(self.conv1(screen)))
        a = t.relu(self.fc1(a.flatten(start_dim=1)))
        return self.fc2(a)


# utility class used to store history
class History:
    def __init__(self, history_depth):
        self._history = [t.zeros([1, 84, 84]) for _ in range(history_depth)]

    def append(self, state: np.ndarray):
        self._history.append(t.tensor(state, dtype=t.float32))
        self._history.pop(0)
        return self

    def get(self):
        # size: (1, history_depth, 84, 84)
        return t.cat(self._history, dim=0).unsqueeze(0)


if __name__ == "__main__":
    q_net = QNet(history_depth).to("cuda:0")
    q_net_t = QNet(history_depth).to("cuda:0")

    # Note:
    # replay_size = 50000 requires about 15GiB memory for replay buffer
    dqn = DQN(q_net, q_net_t,
              t.optim.Adam,
              nn.MSELoss(reduction='sum'),
              replay_size=50000)

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        history = History(history_depth)
        state = history.append(resize_and_convert(env.reset())).get()

        while not terminal:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = dqn.act_discrete_with_noise(
                    {"screen": old_state}
                )

                # info is {"ale.lives": self.ale.lives()}, not used here
                state, reward, terminal, _ = env.step(action.item())
                state = history.append(resize_and_convert(state)).get()
                total_reward += reward

                dqn.store_transition({
                    "state": {"screen": old_state},
                    "action": {"action": action},
                    "next_state": {"screen": state},
                    "reward": reward,
                    "terminal": terminal
                })

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                dqn.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)

        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))
