import time
import random
import inspect
import numpy as np
import torch
import torch.nn as nn

from .base import TorchFramework
from .ddpg_td3 import DDPG_TD3, soft_update
from typing import Union, List, Tuple

from utils.visualize import visualize_graph


class HDDPG_TD3(DDPG_TD3):
    def __init__(self,
                 actor: nn.Module,
                 actor_target: nn.Module,
                 critic: nn.Module,
                 critic_target: nn.Module,
                 critic2: nn.Module,
                 critic2_target: nn.Module,
                 optimizer,
                 criterion,
                 device,
                 q_increase_rate=0.1,
                 q_decrease_rate=0.01,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=1,
                 update_rate=0.005,
                 discount=0.99,
                 replay_size=100000,
                 reward_func=None,
                 policy_noise_func=None,
                 action_trans_func=None):
        """
        Initialize DDPG framework.
        """
        self.q_increase_rate = q_increase_rate
        self.q_decrease_rate = q_decrease_rate
        super(HDDPG_TD3, self).__init__(actor, actor_target, critic, critic_target,
                                        critic2, critic2_target, optimizer, criterion,
                                        device, learning_rate, lr_scheduler, lr_scheduler_params,
                                        batch_size, update_rate, discount, replay_size, reward_func,
                                        policy_noise_func, action_trans_func)

    def update(self, update_value=True, update_policy=True, update_targets=True, concatenate_samples=True):
        """
        Update network weights by sampling from replay buffer.

        Returns:
            (mean value of estimated policy value, value loss)
        """
        batch_size, (state, action, reward, next_state, terminal, *others) = \
            self.rpb.sample_batch(self.batch_size, concatenate_samples, self.device,
                                  sample_keys=["state", "action", "reward", "next_state", "terminal", "*"])

        # Update critic network first
        # Generate value reference :math: `y_i` using target actor and target critic
        with torch.no_grad():
            next_action = self.action_trans_func(self.policy_noise_func(self.act(next_state, True)), next_state)
            next_value = self.criticize(next_state, next_action, True)
            next_value2 = self.criticize2(next_state, next_action, True)
            next_value = torch.min(next_value, next_value2)
            next_value = next_value.view(batch_size, -1)
            y_i = self.reward_func(reward, self.discount, next_value, terminal, others)

        cur_value = self.criticize(state, action)
        cur_value2 = self.criticize2(state, action)

        value_diff = y_i - cur_value
        value_diff2 = y_i - cur_value2
        value_change = torch.where(value_diff > 0,
                                   value_diff * self.q_increase_rate,
                                   value_diff * self.q_decrease_rate)
        value_change2 = torch.where(value_diff2 > 0,
                                    value_diff2 * self.q_increase_rate,
                                    value_diff2 * self.q_decrease_rate)

        value_loss = self.criterion(cur_value, (cur_value + value_change).detach())
        value_loss2 = self.criterion(cur_value2, (cur_value2 + value_change2).detach())

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()
            self.critic2.zero_grad()
            value_loss2.backward()
            self.critic2_optim.step()

        # Update actor network
        cur_action = self.action_trans_func(self.act(state), state)
        act_value = self.criticize(state, cur_action)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            self.actor_optim.step()

        # Update target networks
        if update_targets:
            soft_update(self.actor_target, self.actor, self.update_rate)
            soft_update(self.critic_target, self.critic, self.update_rate)
            soft_update(self.critic2_target, self.critic2, self.update_rate)

        # use .item() to prevent memory leakage
        return -act_policy_loss.item(), (value_loss.item() + value_loss2.item()) / 2
