import torch as t
import numpy as np
from typing import Union, List, Any


class SwarmAgent:
    def __init__(self, base_actor, negotiator,
                 action_dim, observation_dim, history_depth,
                 mean_anneal=0.8, theta_anneal=0.05, batch_size=1,
                 contiguous=True, device="cuda:0"):
        self.actor = base_actor
        self.negotiator = negotiator

        self.action = None
        self.last_action = None
        self.org_action = None
        self.neighbor_action = None
        self.neighbor_action_all = []
        self.observation = None
        self.neighbor_observation = None
        self.reward = None
        self.history = []
        self.history_time_steps = []
        self.neighbors = []  # sorted list
        self.negotiate_rate = 1
        self.negotiate_rate_all = []

        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.history_depth = history_depth
        self.mean_anneal = mean_anneal
        self.theta_anneal = theta_anneal
        self.device = device
        self.batch_size = batch_size
        self.contiguous = contiguous

    def set_neighbors(self, neighbors: List[Any]):
        # neighbors: a list of SwarmAgent class objects
        self.neighbors = neighbors

    def set_observe(self, observation: t.Tensor):
        # observation could be
        # a tensor of dim (observe_dim)
        # or a tensor of dim (B, observe_dim)
        if len(observation.shape) == 1:
            self.observation = observation.unsqueeze(dim=0).to(self.device)
        else:
            self.observation = observation.to(self.device)

    def set_reward(self, reward: Union[float, t.Tensor]):
        # reward could be a float,
        # or tensor of dim (reward_dims),
        # or tensor of dim (B, reward_dims)
        if t.is_tensor(reward):
            if reward.dim != 2:
                reward = reward.view(self.batch_size, -1)
        else:
            if self.batch_size != 1:
                raise RuntimeError("Check batch size!")
            reward = t.tensor(reward).view(1, 1).to(self.device)
        self.reward = reward

    def update_history(self, time_step: int):
        if self.history_depth == 0:
            return
        if len(self.history) == self.history_depth:
            self.history.pop(0)
            self.history_time_steps.pop(0)

        # dim (B, neighbor_num+1, observation_dim)
        full_observation = t.stack([self.observation] + [n.observation for n in self.neighbors], dim=1)

        # dim (B, neighbor_num+1, action_dim)
        full_action = t.stack([self.action] + [n.action for n in self.neighbors], dim=1)

        # dim (B, neighbor_num+1, reward_dim)
        full_rewards = t.stack([self.reward] + [n.reward for n in self.neighbors], dim=1)

        # dim (B, neighbor_num+1, observation_dim + action_dim + reward_dim)
        self.history.append(
            t.cat((full_observation, full_action, full_rewards), dim=2)
        )

        # dim (B, neighbor_num+1)
        self.history_time_steps.append(
            t.full([self.batch_size, len(self.neighbors) + 1],
                   time_step, dtype=t.int32, device=self.device)
        )

    def get_action(self, sync=True):
        if sync:
            return self.last_action
        else:
            return self.action

    def get_negotiation_rate(self):
        return self.negotiate_rate

    def get_history_as_tensor(self):
        # suppose in each history record current agent has n_1, n_2, n_3... neighbors
        # let length = (n_1+1) + (n_2+1) + (n_3+1) + ...
        if len(self.history) > 0:
            # dim (B, length, observation_dim + action_dim + reward_dim)
            return t.cat(self.history, dim=1)
        else:
            return None

    def get_history_time_steps(self):
        # suppose each history record has n_1, n_2, n_3... neighbors
        # let length = (n_1+1) + (n_2+1) + (n_3+1) + ...
        if len(self.history) > 0:
            # dim (B, length)
            return t.cat(self.history_time_steps, dim=1)
        else:
            return None

    def get_sample(self):
        return self.get_history_as_tensor(), \
               self.get_history_time_steps(), \
               self.observation, \
               self.neighbor_observation, \
               self.neighbor_action_all, \
               self.negotiate_rate_all

    def reset(self):
        self.action = None
        self.last_action = None
        self.org_action = None
        self.neighbor_action = None
        self.neighbor_action_all = []
        self.observation = None
        self.neighbor_observation = None
        self.history = []
        self.history_time_steps = []
        self.neighbors = []
        self.negotiate_rate = 1
        self.negotiate_rate_all = []

    def reset_negotiate(self):
        self.negotiate_rate = 1
        # assign a new list, must not clear the old one
        # (which should be stored in replay buffer)
        self.negotiate_rate_all = []
        self.neighbor_action_all = []

    def act_step(self, time_step):
        # dim (B, neighbor_num, observation_dim)
        if len(self.neighbors) > 0:
            self.neighbor_observation = t.stack([n.observation for n in self.neighbors], dim=1)
        else:
            self.neighbor_observation = None

        # generate action with actor
        # dim (B, action_dim)
        self.org_action = self.actor(self.observation, self.neighbor_observation,
                                     self.get_history_as_tensor(), self.get_history_time_steps(),
                                     time_step)
        self.last_action = self.action = self.org_action
        return self.action

    def negotiate_step(self, time_step):
        self.last_action = self.action
        # dim (B, neighbor_num, observation_dim)
        if len(self.neighbors) > 0:
            self.neighbor_observation = t.stack([n.observation for n in self.neighbors], dim=1)
        else:
            self.neighbor_observation = None

        self.neighbor_action_all.append(self.neighbor_action)

        change = self.negotiator(self.observation, self.neighbor_observation,
                                 self.action, self.neighbor_action,
                                 self.get_history_as_tensor(), self.get_history_time_steps(),
                                 time_step, self.negotiate_rate)

        # dim (B, action_dim)
        if self.contiguous:
            self.action = self.action + self.negotiate_rate * change
        else:
            self.action = (self.action + self.negotiate_rate * change) / (1 + self.negotiate_rate)

        self.negotiate_rate_all.append(self.negotiate_rate)
        self._update_negotiate_rate()
        return self.action

    def final_step(self):
        # dim (B, neighbor_num, action_dim)
        if len(self.neighbors) > 0:
            self.neighbor_observation = t.stack([n.observation for n in self.neighbors], dim=1)
        else:
            self.neighbor_observation = None

        self.neighbor_action_all.append(self.neighbor_action)
        return self.action

    def _update_negotiate_rate(self):
        self.negotiate_rate *= np.random.normal(self.mean_anneal, self.theta_anneal, 1)[0]