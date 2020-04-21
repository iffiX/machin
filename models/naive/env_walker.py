import torch as t
import torch.nn as nn


class Actor(nn.Module):
    # naive actor for env.walker
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.max_action = max_action

    def forward(self, state):
        a = self.prelu1(self.fc1(state))
        a = self.prelu2(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):
    # naive actor for env.walker
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = self.prelu1(self.fc1(state_action))
        q = self.prelu2(self.fc2(q))
        q = self.fc3(q)
        return q