from machin.env.utils.openai_gym import disable_view_window
from machin.frame.algorithms import DQNPer
from machin.utils.logging import default_logger as logger

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


# Q network model definition
# for atari games
class RecurrentQNet(nn.Module):
    def __init__(self, action_num):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            action_num: (int): write your description
        """
        super(RecurrentQNet, self).__init__()
        self.gru = nn.GRU(128, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, action_num)

    def forward(self, mem=None, hidden=None, history_mem=None):
        """
        Perform forward forward : forward

        Args:
            self: (todo): write your description
            mem: (todo): write your description
            hidden: (todo): write your description
            history_mem: (todo): write your description
        """
        if mem is not None:
            # in sampling
            a, h = self.gru(mem.unsqueeze(1), hidden)
            return self.fc2(t.relu(
                self.fc1(t.relu(
                    a.flatten(start_dim=1)
                ))
            )), h
        else:
            # in updating
            batch_size = history_mem.shape[0]
            seq_length = history_mem.shape[1]
            hidden = t.zeros([1, batch_size, 256],
                             device=history_mem.device)
            for i in range(seq_length):
                _, hidden = self.gru(history_mem[:, i].unsqueeze(1), hidden)
            # a[:, -1] = h
            return self.fc2(t.relu(
                self.fc1(t.relu(
                    hidden.transpose(0, 1).flatten(start_dim=1)
                ))
            ))


if __name__ == "__main__":
    r_q_net = RecurrentQNet(action_num).to("cuda:0")
    r_q_net_t = RecurrentQNet(action_num).to("cuda:0")

    drqn = DQNPer(r_q_net, r_q_net_t,
                  t.optim.Adam,
                  nn.MSELoss(reduction='sum'),
                  learning_rate=5e-4)

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        hidden = t.zeros([1, 1, 256])
        state = convert(env.reset())
        history = History(history_depth, (1, 128))

        while not terminal:
            step += 1
            with t.no_grad():
                old_state = state
                history.append(state)
                # agent model inference
                action, hidden = drqn.act_discrete_with_noise(
                    {"mem": old_state, "hidden": hidden}
                )

                # info is {"ale.lives": self.ale.lives()}, not used here
                state, reward, terminal, _ = env.step(action.item())
                state = convert(state)
                total_reward += reward

                # history mem includes current state
                old_history = history.get()
                new_history = history.append(state).get()
                drqn.store_transition({
                    "state": {"history_mem": old_history},
                    "action": {"action": action},
                    "next_state": {"history_mem": new_history},
                    "reward": reward,
                    "terminal": terminal
                })

        # update, update more if episode is longer, else less
        if episode > 20:
            for _ in range(step // 10):
                drqn.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)

        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))
