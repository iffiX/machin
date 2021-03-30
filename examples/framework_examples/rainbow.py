from machin.frame.algorithms import RAINBOW
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym

# configurations
env = gym.make("CartPole-v0")
observe_dim = 4
action_num = 2
# maximum and minimum of reward value
# since reward is 1 for every step, maximum q value should be
# below 20(reward_future_steps) * (1 + discount ** n_steps) < 40
value_max = 40
value_min = 0
reward_future_steps = 20
max_episodes = 1000
max_steps = 200
solved_reward = 190
solved_repeat = 5


# model definition
class QNet(nn.Module):
    # this test setup lacks the noisy linear layer and dueling structure.
    def __init__(self, state_dim, action_num, atom_num=10):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num * atom_num)
        self.action_num = action_num
        self.atom_num = atom_num

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return t.softmax(self.fc3(a).view(-1, self.action_num, self.atom_num), dim=-1)


if __name__ == "__main__":
    q_net = QNet(observe_dim, action_num)
    q_net_t = QNet(observe_dim, action_num)

    rainbow = RAINBOW(
        q_net,
        q_net_t,
        t.optim.Adam,
        value_min,
        value_max,
        reward_future_steps=reward_future_steps,
    )

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

        tmp_observations = []
        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = rainbow.act_discrete_with_noise({"state": old_state})
                state, reward, terminal, _ = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward

                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )

        rainbow.store_episode(tmp_observations)

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                rainbow.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0
