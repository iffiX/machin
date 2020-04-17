import torch as t
import numpy as np


class SwarmAgent:
    def __init__(self, base_actor, negotiator, neighbor_num,
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
        if history_depth > 0:
            self.history = [t.zeros([batch_size, neighbor_num + 1, observation_dim + action_dim + 1],
                                    dtype=t.float32, device=device)]
        else:
            self.history = None
        self.neighbors = []  # sorted list
        self.negotiate_rate = 1
        self.negotiate_rate_all = []

        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.history_depth = history_depth
        self.mean_anneal = mean_anneal
        self.theta_anneal = theta_anneal
        self.neighbor_num = neighbor_num  # maximum neighbor num
        self.device = device
        self.batch_size = batch_size
        self.contiguous = contiguous

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_observe(self, observation: t.Tensor):
        self.observation = t.flatten(observation.to(self.device), start_dim=1)

    def set_reward(self, reward):
        self.reward = reward

    def update_history(self):
        if self.history_depth == 0:
            return
        if len(self.history) == self.history_depth:
            self.history.pop(0)

        pad_num = self.neighbor_num - len(self.neighbors)
        full_observation = t.stack([self.observation] + [n.observation for n in self.neighbors] +
                                   [t.zeros_like(self.observation)] * pad_num,
                                   dim=1)
        full_action = t.stack([self.action] + [n.action for n in self.neighbors] +
                              [t.zeros_like(self.action)] * pad_num, dim=1)
        full_rewards = t.stack([self.reward] + [n.reward for n in self.neighbors] +
                               [t.zeros_like(self.reward)] * pad_num, dim=1).unsqueeze(dim=-1)
        self.history.append(t.cat((full_observation, full_action, full_rewards), dim=2))

    def get_action(self, sync=True):
        if sync:
            return self.last_action
        else:
            return self.action

    def get_negotiation_rate(self):
        return self.negotiate_rate

    def get_history_as_tensor(self):
        return t.stack(self.history, dim=1)

    def get_sample(self):
        return self.get_history_as_tensor() if self.history_depth > 0 else None, \
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
        if self.history_depth > 0:
            self.history = [t.zeros([self.batch_size,
                                     self.neighbor_num + 1,
                                     self.observation_dim + self.action_dim + 1],
                                    dtype=t.float32, device=self.device)]
        else:
            self.history = None
        self.neighbors = []
        self.negotiate_rate = 1
        self.negotiate_rate_all = []

    def reset_negotiate(self):
        self.negotiate_rate = 1
        # assign a new list, must not clear the old one
        # (which should be stored in replay buffer)
        self.negotiate_rate_all = []
        self.neighbor_action_all = []

    def act_step(self):
        pad_num = self.neighbor_num - len(self.neighbors)
        self.neighbor_observation = t.stack([n.observation for n in self.neighbors] +
                                            [t.zeros_like(self.observation)] * pad_num, dim=1)  # (B, N, O)
        # generate action with actor
        self.org_action = self.actor(self.observation, self.neighbor_observation,  # (B, A)
                                     self.get_history_as_tensor())
        self.last_action = self.action = self.org_action
        return self.action

    def negotiate_step(self):
        pad_num = self.neighbor_num - len(self.neighbors)
        self.last_action = self.action
        self.neighbor_action = t.stack([n.get_action() for n in self.neighbors] +
                                       [t.zeros_like(self.action)] * pad_num, dim=1)  # (B, N, A)
        self.neighbor_action_all.append(self.neighbor_action)
        change = self.negotiator(self.observation, self.neighbor_observation,
                                 self.get_history_as_tensor(),
                                 self.action, self.neighbor_action,
                                 self.negotiate_rate)

        if self.contiguous:
            self.action = self.action + self.negotiate_rate * change
        else:
            self.action = (self.action + self.negotiate_rate * change) / (1 + self.negotiate_rate)

        self.negotiate_rate_all.append(self.negotiate_rate)
        self._update_negotiate_rate()
        return self.action

    def final_step(self):
        pad_num = self.neighbor_num - len(self.neighbors)
        self.neighbor_action = t.stack([n.get_action() for n in self.neighbors] +
                                       [t.zeros_like(self.action)] * pad_num, dim=1)  # (B, N, A)
        self.neighbor_action_all.append(self.neighbor_action)
        return self.action

    def _update_negotiate_rate(self):
        self.negotiate_rate *= np.random.normal(self.mean_anneal, self.theta_anneal, 1)[0]