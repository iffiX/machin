from machin.frame.algorithms import PPO, GAIL
from machin.utils.logging import default_logger as logger
from torch.distributions import Categorical
import torch as t
import torch.nn as nn
import gym

# configurations
env = gym.make("CartPole-v0")
observe_dim = 4
action_num = 2
max_episodes = 1000
expert_episodes = 100
max_steps = 200
solved_reward = 190
solved_repeat = 5


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + 1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.action_num = action_num

    def forward(self, state, action: t.Tensor):
        d = t.relu(
            self.fc1(
                t.cat(
                    [state, action.type_as(state).view(-1, 1) / self.action_num], dim=1
                )
            )
        )
        d = t.relu(self.fc2(d))
        d = t.sigmoid(self.fc3(d))
        return d


def run_episode(ppo, env):
    total_reward = 0
    terminal = False
    step = 0
    state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

    observations = []
    while not terminal and step <= max_steps:
        step += 1
        with t.no_grad():
            old_state = state
            # agent model inference
            action = ppo.act({"state": old_state})[0]
            state, reward, terminal, _ = env.step(action.item())
            state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
            total_reward += reward

            observations.append(
                {
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps,
                }
            )
    return observations, total_reward


def generate_expert_episodes():
    actor = Actor(observe_dim, action_num)
    critic = Critic(observe_dim)

    ppo = PPO(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"))
    logger.info("Training expert PPO")

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        # update
        episode_observations, episode_total_reward = run_episode(ppo, env)
        ppo.store_episode(episode_observations)
        ppo.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + episode_total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                break
        else:
            reward_fulfilled = 0

    trajectories = []
    for i in range(expert_episodes):
        logger.info(f"Generating trajectory {i}")
        trajectories.append(
            [
                {"state": s["state"], "action": s["action"]}
                for s in run_episode(ppo, env)[0]
            ]
        )
    return trajectories


if __name__ == "__main__":
    actor = Actor(observe_dim, action_num)
    critic = Critic(observe_dim)
    discriminator = Discriminator(observe_dim, action_num)
    ppo = PPO(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"))
    gail = GAIL(discriminator, ppo, t.optim.Adam)

    for expert_episode in generate_expert_episodes():
        gail.store_expert_episode(expert_episode)

    # begin training
    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    terminal = False
    logger.info("Training GAIL")

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
                action = gail.act({"state": old_state})[0]
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

        # update
        gail.store_episode(tmp_observations)
        gail.update()

        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                break
        else:
            reward_fulfilled = 0
