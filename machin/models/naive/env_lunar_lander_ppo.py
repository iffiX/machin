import torch as t
import torch.nn as nn

from torch.distributions import Categorical


class Actor(nn.Module):
    # naive actor for lunar-lander-v2
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.softmax(self.fc3(a), dim=1)

        a_dist = Categorical(probs=a)
        action = action.squeeze(1) if action is not None else a_dist.sample()
        entropy = a_dist.entropy()
        return action.unsqueeze(1).detach(), \
               a_dist.log_prob(action).unsqueeze(1), \
               entropy.unsqueeze(1)


class Critic(nn.Module):
    # naive actor for lunar-lander-v2
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        q = t.relu(self.fc1(state))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q