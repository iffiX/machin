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
class QNet(nn.Module):
    def __init__(self, history_depth, action_num):
        super().__init__()
        self.fc1 = nn.Linear(128 * history_depth, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_num)

    def forward(self, mem):
        return self.fc3(t.relu(self.fc2(t.relu(self.fc1(mem.flatten(start_dim=1))))))


if __name__ == "__main__":
    q_net = QNet(history_depth, action_num).to("cuda:0")
    q_net_t = QNet(history_depth, action_num).to("cuda:0")

    dqn = DQNPer(
        q_net, q_net_t, t.optim.Adam, nn.MSELoss(reduction="sum"), learning_rate=5e-4
    )

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = convert(env.reset())
        history = History(history_depth, (1, 128))

        while not terminal:
            step += 1
            with t.no_grad():
                history.append(state)
                # agent model inference
                action = dqn.act_discrete_with_noise({"mem": history.get()})

                # info is {"ale.lives": self.ale.lives()}, not used here
                state, reward, terminal, _ = env.step(action.item())
                state = convert(state)
                total_reward += reward
                old_history = history.get()
                new_history = history.append(state).get()
                dqn.store_transition(
                    {
                        "state": {"mem": old_history},
                        "action": {"action": action},
                        "next_state": {"mem": new_history},
                        "reward": reward,
                        "terminal": terminal,
                    }
                )

        # update, update more if episode is longer, else less
        if episode > 20:
            for _ in range(step // 10):
                dqn.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1

        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")
