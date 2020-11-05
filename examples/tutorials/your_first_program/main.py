from machin.frame.algorithms import DQN
from machin.utils.logging import default_logger as logger
from machin.model.nets import static_module_wrapper, dynamic_module_wrapper
import torch as t
import torch.nn as nn
import gym

# configurations
env = gym.make("CartPole-v0")
observe_dim = 4
action_num = 2
max_episodes = 1000
max_steps = 200
solved_reward = 190
solved_repeat = 5


# model definition
class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        """
        Initialize the internal state.

        Args:
            self: (todo): write your description
            state_dim: (int): write your description
            action_num: (int): write your description
        """
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, some_state):
        """
        Forward forward forward algorithm.

        Args:
            self: (todo): write your description
            some_state: (todo): write your description
        """
        a = t.relu(self.fc1(some_state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


if __name__ == "__main__":
    # let framework determine input/output device based on parameter location
    # a warning will be thrown.
    q_net = QNet(observe_dim, action_num)
    q_net_t = QNet(observe_dim, action_num)

    # to mark the input/output device Manually
    # will not work if you move your model to other devices
    # after wrapping

    # q_net = static_module_wrapper(q_net, "cpu", "cpu")
    # q_net_t = static_module_wrapper(q_net_t, "cpu", "cpu")

    # to mark the input/output device Automatically
    # will not work if you model locates on multiple devices

    # q_net = dynamic_module_wrapper(q_net)
    # q_net_t = dynamic_module_wrapper(q_net_t)

    dqn = DQN(q_net, q_net_t,
              t.optim.Adam,
              nn.MSELoss(reduction='sum'))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = dqn.act_discrete_with_noise(
                    {"some_state": old_state}
                )
                state, reward, terminal, _ = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward

                dqn.store_transition({
                    "state": {"some_state": old_state},
                    "action": {"action": action},
                    "next_state": {"some_state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps
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

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0
