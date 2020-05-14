import torch as t
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import MultivariateNormal


class Actor(nn.Module):
    # naive actor for env.walker
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128, action_dim)
        self.fc_sigma = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))

        # mu = self.max_action * t.tanh(self.fc_mu(a))

        mu = self.fc_mu(a)

        # we assume that each dimension of your action is not correlated
        # therefore the covariance matrix is a positive definite diagonal matrix

        # static, preset standard error
        # diag = t.full(mu.shape, 0.5, device=mu.device)

        # dynamic, trainable standard error
        diag = softplus(self.fc_sigma(a))
        cov = t.diag_embed(diag)
        a_dist = MultivariateNormal(mu, cov)
        action = action if action is not None else a_dist.sample()
        action_log_prob = a_dist.log_prob(action)
        entropy = a_dist.entropy()
        return action.detach(), action_log_prob.unsqueeze(1), entropy.mean()


class Critic(nn.Module):
    # naive actor for env.walker
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        q = t.relu(self.fc1(state))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q