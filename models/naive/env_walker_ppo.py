import torch as t
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import MultivariateNormal


# class Actor(nn.Module):
#     # naive actor for env.walker
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()
#
#         self.fc1 = nn.Linear(state_dim, 400)
#         self.fc2 = nn.Linear(400, 300)
#         self.fc_mu = nn.Linear(300, action_dim)
#         self.fc_sigma = nn.Linear(300, action_dim)
#         self.max_action = max_action
#
#     def forward(self, state):
#         a = t.relu(self.fc1(state))
#         a = t.relu(self.fc2(a))
#
#         mu = self.max_action * t.tanh(self.fc_mu(a))
#         sigma = softplus(self.fc_sigma(a))
#
#         # we assume that each dimension of your action is not correlated
#         # therefore the covariance matrix is a positive definite diagonal matrix
#         diag = sigma ** 2
#         cov = t.diag_embed(diag)
#         dist = MultivariateNormal(mu, cov)
#         action = dist.sample()
#         action_log_prob = dist.log_prob(action)
#         action = action.clamp(-self.max_action, self.max_action)
#         return action.detach(), action_log_prob

class Actor(nn.Module):
    # naive actor for env.walker
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mu = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))

        mu = self.max_action * t.tanh(self.fc_mu(a))

        # we assume that each dimension of your action is not correlated
        # therefore the covariance matrix is a positive definite diagonal matrix
        diag = t.full(mu.shape, 0.2, device=mu.device)
        cov = t.diag_embed(diag)
        dist = MultivariateNormal(mu, cov)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-self.max_action, self.max_action)
        return action.detach(), action_log_prob


class Critic(nn.Module):
    # naive actor for env.walker
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state):
        q = t.relu(self.fc1(state))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q