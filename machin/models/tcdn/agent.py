import torch as t
import numpy as np
from typing import Union, List, Any
from utils.checker import check_shape

# class SwarmAgent:
#     def __init__(self, base_actor, negotiator,
#                  action_dim, observation_dim, history_depth,
#                  mean_anneal=0.8, theta_anneal=0.05,
#                  contiguous=True, device="cuda:0"):
#         self.actor = base_actor
#         self.negotiator = negotiator
#
#         self.action = None
#         self.last_action = None
#         self.org_action = None
#         self.neighbor_action = None
#         self.neighbor_action_all = []
#         self.observation = None
#         self.neighbor_observation = None
#         self.reward = None
#         self.history = []
#         self.history_time_steps = []
#         self.neighbors = []  # sorted list
#         self.negotiate_rate = 1
#         self.negotiate_rate_all = []
#
#         self.action_dim = action_dim
#         self.observation_dim = observation_dim
#         self.history_depth = history_depth
#         self.mean_anneal = mean_anneal
#         self.theta_anneal = theta_anneal
#         self.device = device
#         self.contiguous = contiguous
#
#     def set_neighbors(self, neighbors: List[Any]):
#         # neighbors: a list of SwarmAgent class objects
#         self.neighbors = neighbors
#
#     def set_observe(self, observation: t.Tensor):
#         # observation could be
#         # a tensor of dim (observe_dim)
#         # or a tensor of dim (1, observe_dim)
#         if len(observation.shape) == 1:
#             self.observation = observation.unsqueeze(dim=0).to(self.device)
#         else:
#             self.observation = observation.to(self.device)
#
#     def set_reward(self, reward: Union[float, t.Tensor]):
#         # reward could be a float,
#         # or tensor of dim (reward_dims),
#         # or tensor of dim (1, reward_dims)
#         if t.is_tensor(reward):
#             if reward.dim != 2:
#                 reward = reward.view(1, -1)
#         else:
#             reward = t.tensor(reward).view(1, 1).to(self.device)
#         self.reward = reward
#
#     def update_history(self, time_step: int):
#         if self.history_depth == 0:
#             return
#         if len(self.history) == self.history_depth:
#             self.history.pop(0)
#             self.history_time_steps.pop(0)
#
#         # dim (1, neighbor_num+1, observation_dim)
#         full_observation = t.stack([self.observation] + [n.observation for n in self.neighbors], dim=1)
#
#         # dim (1, neighbor_num+1, action_dim)
#         full_action = t.stack([self.action] + [n.action for n in self.neighbors], dim=1)
#
#         # dim (1, neighbor_num+1, reward_dim)
#         full_rewards = t.stack([self.reward] + [n.reward for n in self.neighbors], dim=1)
#
#         # dim (1, neighbor_num+1, observation_dim + action_dim + reward_dim)
#         self.history.append(
#             t.cat((full_observation, full_action, full_rewards), dim=2)
#         )
#
#         # dim (1, neighbor_num+1)
#         self.history_time_steps.append(
#             t.full([1, len(self.neighbors) + 1],
#                    time_step, dtype=t.int32, device=self.device)
#         )
#
#     def get_action(self, sync=True):
#         if sync:
#             return self.last_action
#         else:
#             return self.action
#
#     def get_history_as_tensor(self):
#         # suppose in each history record current agent has n_1, n_2, n_3... neighbors
#         # let length = (n_1+1) + (n_2+1) + (n_3+1) + ...
#         if len(self.history) > 0:
#             # dim (1, length, observation_dim + action_dim + reward_dim)
#             return t.cat(self.history, dim=1)
#         else:
#             return None
#
#     def get_history_time_steps(self):
#         # suppose each history record has n_1, n_2, n_3... neighbors
#         # let length = (n_1+1) + (n_2+1) + (n_3+1) + ...
#         if len(self.history) > 0:
#             # dim (1, length)
#             return t.cat(self.history_time_steps, dim=1)
#         else:
#             return None
#
#     def get_sample(self):
#         return self.get_history_as_tensor(), \
#                self.get_history_time_steps(), \
#                self.observation, \
#                self.neighbor_observation, \
#                self.neighbor_action_all, \
#                self.negotiate_rate_all
#
#     def reset(self):
#         self.action = None
#         self.last_action = None
#         self.org_action = None
#         self.neighbor_action = None
#         self.neighbor_action_all = []
#         self.observation = None
#         self.neighbor_observation = None
#         self.history = []
#         self.history_time_steps = []
#         self.neighbors = []
#         self.negotiate_rate = 1
#         self.negotiate_rate_all = []
#
#     def reset_negotiate(self):
#         self.negotiate_rate = 1
#         # assign a new list, must not clear the old one
#         # (which should be stored in replay buffer)
#         self.negotiate_rate_all = []
#         self.neighbor_action_all = []
#
#     def act_step(self, time_step):
#         # dim (1, neighbor_num, observation_dim)
#         if len(self.neighbors) > 0:
#             self.neighbor_observation = t.stack([n.observation for n in self.neighbors], dim=1)
#         else:
#             self.neighbor_observation = None
#
#         # generate action with actor
#         # dim (1, action_dim)
#         self.org_action = self.actor(self.observation, self.neighbor_observation,
#                                      self.get_history_as_tensor(), self.get_history_time_steps(),
#                                      time_step)
#         self.last_action = self.action = self.org_action
#         return self.action
#
#     def negotiate_step(self, time_step):
#         self.last_action = self.action
#         # dim (1, neighbor_num, observation_dim)
#         if len(self.neighbors) > 0:
#             self.neighbor_action = t.stack([n.action for n in self.neighbors], dim=1)
#         else:
#             self.neighbor_action = None
#
#         self.neighbor_action_all.append(self.neighbor_action)
#
#         change = self.negotiator(self.observation, self.action,
#                                  self.neighbor_observation, self.neighbor_action,
#                                  self.get_history_as_tensor(), self.get_history_time_steps(),
#                                  time_step, self.negotiate_rate)
#
#         # dim (1, action_dim)
#         if self.contiguous:
#             self.action = self.action + self.negotiate_rate * change
#         else:
#             self.action = (self.action + self.negotiate_rate * change) / (1 + self.negotiate_rate)
#
#         self.negotiate_rate_all.append(self.negotiate_rate)
#         self._update_negotiate_rate()
#         return self.action
#
#     def final_step(self):
#         # dim (1, neighbor_num, action_dim)
#         if len(self.neighbors) > 0:
#             self.neighbor_observation = t.stack([n.observation for n in self.neighbors], dim=1)
#         else:
#             self.neighbor_observation = None
#
#         self.neighbor_action_all.append(self.neighbor_action)
#         return self.action
#
#     def _update_negotiate_rate(self):
#         self.negotiate_rate *= np.clip(np.random.normal(self.mean_anneal, self.theta_anneal, 1), 0, 1)[0]


class SwarmAgent:
    def __init__(self, base_actor, negotiator, neighbor_num,
                 action_dim, observation_dim,  history_depth,
                 mean_anneal=0.8, theta_anneal=0.05,
                 batch_size=1, reward_dim=1,
                 contiguous=True, device="cuda:0"):
        self.actor = base_actor
        self.negotiator = negotiator
        self.neighbor_num = neighbor_num

        self.action = None
        self.last_action = None
        self.org_action = None
        self.neighbor_action = None
        self.neighbor_action_all = []
        self.observation = None
        self.neighbor_observation = None
        self.time_step = None
        self.reward = None
        self.history = []
        self.history_time_steps = []
        self.neighbors = []  # sorted list
        self.negotiate_rate = t.ones([batch_size, 1], dtype=t.float, device=device)
        self.negotiate_rate_all = []

        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.reward_dim = reward_dim
        self.history_dim = action_dim + observation_dim + reward_dim
        self.history_depth = history_depth
        self.mean_anneal = mean_anneal
        self.theta_anneal = theta_anneal
        self.device = device
        self.batch_size = batch_size
        self.contiguous = contiguous

    def set_neighbors(self, neighbors: List[Any]):
        # neighbors: a list of SwarmAgent class objects
        if len(neighbors) > self.neighbor_num:
            raise RuntimeError("Too many neighbors")
        self.neighbors = neighbors

    def set_observe(self, observation: t.Tensor):
        # observation should be a tensor of dim (B, observe_dim)
        check_shape(observation, [self.batch_size, self.observation_dim])
        self.observation = observation.to(self.device)

    def set_reward(self, reward: t.Tensor):
        # reward should be a tensor of dim (B, reward_dim),
        check_shape(reward, [self.batch_size, self.reward_dim])
        self.reward = reward

    def update_history(self, time_step: int):
        if self.history_depth == 0:
            return
        if len(self.history) == self.history_depth:
            self.history.pop(0)
            self.history_time_steps.pop(0)

        pad_num = self.neighbor_num - len(self.neighbors)

        # dim (B, neighbor_num+1, observation_dim)
        full_observation = t.stack([self.observation] + [n.observation for n in self.neighbors] +
                                   [t.zeros_like(self.observation)] * pad_num, dim=1)

        # dim (B, neighbor_num+1, action_dim)
        full_action = t.stack([self.action] + [n.action for n in self.neighbors] +
                              [t.zeros_like(self.action)] * pad_num, dim=1)

        # dim (B, neighbor_num+1, reward_dim)
        full_rewards = t.stack([self.reward] + [n.reward for n in self.neighbors] +
                               [t.zeros_like(self.reward)] * pad_num, dim=1)

        # dim (B, neighbor_num+1, observation_dim + action_dim + reward_dim)
        self.history.append(
            t.cat((full_observation, full_action, full_rewards), dim=2)
        )

        # dim (B, neighbor_num+1)
        self.history_time_steps.append(
            t.full([self.batch_size, self.neighbor_num + 1],
                   time_step, dtype=t.int32, device=self.device)
        )

    def get_sample(self):
        if self.observation is None or self.neighbor_observation is None:
            raise RuntimeError("Run an act_step before sampling.")
        return self._get_history_as_tensor(), \
               self._get_history_time_steps_as_tensor(), \
               self.observation, \
               self.neighbor_observation, \
               self._get_neighbor_action_all_as_tensor(), \
               self._get_negotiate_rate_all_as_tensor(), \
               self.time_step

    def reset(self):
        self.action = None
        self.last_action = None
        self.org_action = None
        self.neighbor_action = None
        self.neighbor_action_all = []
        self.observation = None
        self.neighbor_observation = None
        self.time_step = None
        self.history = []
        self.history_time_steps = []
        self.neighbors = []
        self.negotiate_rate = t.ones([self.batch_size, 1], dtype=t.float, device=self.device)
        self.negotiate_rate_all = []

    def reset_negotiate(self):
        self.negotiate_rate = t.ones([self.batch_size, 1], dtype=t.float, device=self.device)
        # assign a new list, must not clear the old one
        # (which should be stored in replay buffer)
        self.negotiate_rate_all = []
        self.neighbor_action_all = []

    def act_step(self, time_step: int):
        self.time_step = t.full([self.batch_size, self.neighbor_num+1], time_step,
                                 dtype=t.int32, device=self.device)
        pad_num = self.neighbor_num - len(self.neighbors)
        # dim (B, neighbor_num, observation_dim)
        self.neighbor_observation = t.stack([n.observation for n in self.neighbors] +
                                            [t.zeros_like(self.observation)] * pad_num, dim=1)

        # generate action with actor
        # dim (B, action_dim)
        self.org_action = self.actor(self.observation, self.neighbor_observation,
                                     self._get_history_as_tensor(), self._get_history_time_steps_as_tensor(),
                                     self.time_step)
        self.last_action = self.action = self.org_action
        return self.action

    def negotiate_step(self):
        pad_num = self.neighbor_num - len(self.neighbors)
        self.last_action = self.action
        # dim (B, neighbor_num, observation_dim)
        self.neighbor_action = t.stack([n._get_action(sync=True) for n in self.neighbors] +
                                       [t.zeros_like(self.action)] * pad_num, dim=1)


        self.neighbor_action_all.append(self.neighbor_action)

        change = self.negotiator(self.observation, self.action,
                                 self.neighbor_observation, self.neighbor_action,
                                 self._get_history_as_tensor(), self._get_history_time_steps_as_tensor(),
                                 self.time_step,
                                 self.negotiate_rate)

        # dim (B, action_dim)
        self.action = (self.action + self.negotiate_rate * change) / (1 + self.negotiate_rate)

        self.negotiate_rate_all.append(self.negotiate_rate.clone())
        self._update_negotiate_rate()
        return self.action

    def final_step(self):
        pad_num = self.neighbor_num - len(self.neighbors)
        # dim (B, neighbor_num, action_dim)
        self.neighbor_action = t.stack([n._get_action(sync=True) for n in self.neighbors] +
                                       [t.zeros_like(self.action)] * pad_num, dim=1)


        self.neighbor_action_all.append(self.neighbor_action)
        return self.action

    def _get_action(self, sync=True):
        if sync:
            return self.last_action
        else:
            return self.action

    def _get_negotiate_rate_all_as_tensor(self):
        if len(self.negotiate_rate_all) > 0:
            # dim (B, negotiate_rounds)
            return t.cat(self.negotiate_rate_all, dim=1)
        else:
            return t.zeros((self.batch_size, 0), dtype=t.float, device=self.device)

    def _get_history_as_tensor(self):
        # dim (B, (neighbor_num+1) * history_depth, observation_dim + action_dim + reward_dim)
        if self.history_depth > 0:
            pad_num = self.history_depth - len(self.history)
            return t.cat([t.zeros([self.batch_size, pad_num * (self.neighbor_num+1),
                                  self.history_dim], dtype=t.float, device=self.device)] + self.history,
                         dim=1)
        else:
            return t.zeros((self.batch_size, 0, self.history_dim), dtype=t.float, device=self.device)

    def _get_history_time_steps_as_tensor(self):
        if self.history_depth > 0:
            pad_num = self.history_depth - len(self.history)
            # dim (B, (neighbor_num+1) * history_depth)
            return t.cat([t.zeros([self.batch_size, pad_num * (self.neighbor_num + 1)],
                                  dtype=t.int32, device=self.device)] + self.history_time_steps,
                         dim=1)
        else:
            return t.zeros((self.batch_size, 0), dtype=t.int32, device=self.device)

    def _get_neighbor_action_all_as_tensor(self):
        # dim (B, negotiate_rounds+1, neighbor_num, action_dim)
        return t.stack(self.neighbor_action_all, dim=1)

    def _update_negotiate_rate(self):
        std = t.full([self.batch_size, 1], self.theta_anneal, dtype=t.float, device=self.device)
        self.negotiate_rate *= t.clamp(t.normal(self.mean_anneal, std), 0, 1)
