import time
import random
import inspect
import numpy as np
import torch
import torch.nn as nn

from .base import TorchFramework
from .ddpg import soft_update, hard_update, safe_call, ReplayBuffer
from typing import Union, List, Tuple

from utils.visualize import visualize_graph


class DDPG_TD3(TorchFramework):
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
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=1,
                 update_rate=0.005,
                 discount=0.99,
                 replay_size=100000,
                 reward_func=None,
                 action_trans_func=None):
        """
        Initialize DDPG framework.
        """
        self.device = device
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.rpb = ReplayBuffer(replay_size)

        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.critic2 = critic2
        self.critic2_target = critic2_target
        self.actor_optim = optimizer(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), lr=learning_rate)
        self.critic2_optim = optimizer(self.critic2.parameters(), lr=learning_rate)

        # Make sure target and online networks have the same weight
        with torch.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic2, self.critic2_target)

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(self.actor_optim, *lr_scheduler_params[0])
            self.critic_lr_sch = lr_scheduler(self.critic_optim, *lr_scheduler_params[1])
            self.critic2_lr_sch = lr_scheduler(self.critic2_optim, *lr_scheduler_params[1])

        self.criterion = criterion

        self.reward_func = DDPG_TD3.bellman_function if reward_func is None else reward_func
        self.action_trans_func = DDPG_TD3.action_trans_func if action_trans_func is None else action_trans_func

        super(DDPG_TD3, self).__init__()
        self.set_top(["actor", "critic", "critic2", "actor_target", "critic_target", "critic2_target"])
        self.set_restorable(["actor_target", "critic_target", "critic2_target"])

    def act(self, state, use_target=False):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Policy produced by actor.
        """
        if use_target:
            return safe_call(self.actor_target, state)
        else:
            return safe_call(self.actor, state)

    def act_with_noise(self, state, noise_param=(0.0, 1.0), ratio=1.0, mode="uniform", use_target=False):
        """
        Use actor network to give a policy (with uniform noise added) to the current state.

        Args:
            noise_param: A single tuple or a list of tuples specifying noise params
            for each column (last dimension) of action.
        Returns:
            Policy (with uniform noise) produced by actor.
        """
        if mode == "uniform":
            return self.add_uniform_noise_to_action(self.act(state, use_target),
                                                    noise_param, ratio)
        elif mode == "normal":
            return self.add_normal_noise_to_action(self.act(state, use_target),
                                                    noise_param, ratio)
        else:
            raise RuntimeError("Unknown noise type: " + str(mode))

    def add_uniform_noise_to_action(self, action, noise_param=(0.0, 1.0), ratio=1.0):
        if isinstance(noise_param[0], tuple):
            if len(noise_param) != action.shape[-1]:
                raise ValueError("Noise range length doesn't match the last dimension of action")
            noise = torch.rand(action.shape, device=action.device)
            for i in range(action.shape[-1]):
                noi_p = noise_param[i]
                noise.view(-1, noise.shape[-1])[:, i] *= noi_p[1] - noi_p[0]
                noise.view(-1, noise.shape[-1])[:, i] += noi_p[0]
        else:
            noise = torch.rand(action.shape, device=action.device) \
                    * (noise_param[1] - noise_param[0]) + noise_param[0]
        return action + noise * ratio

    def add_normal_noise_to_action(self, action, noise_param=(0.0, 0.1), ratio=1.0):
        if isinstance(noise_param[0], tuple):
            if len(noise_param) != action.shape[-1]:
                raise ValueError("Noise range length doesn't match the last dimension of action")
            noise = torch.randn(action.shape, device=action.device)
            for i in range(action.shape[-1]):
                noi_p = noise_param[i]
                noise.view(-1, noise.shape[-1])[:, i] *= noi_p[1]
                noise.view(-1, noise.shape[-1])[:, i] += noi_p[0]
        else:
            noise = torch.rand(action.shape, device=action.device) \
                    * noise_param[1] + noise_param[0]
        return action + noise * ratio

    def criticize(self, state, action, use_target=False):
        """
        Use the first critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        Notes:
            State and action will be concatenated in dimension 1
        """
        if use_target:
            return safe_call(self.critic_target, state, action)
        else:
            return safe_call(self.critic, state, action)

    def criticize2(self, state, action, use_target=False):
        """
        Use the second critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        Notes:
            State and action will be concatenated in dimension 1
        """
        if use_target:
            return safe_call(self.critic2_target, state, action)
        else:
            return safe_call(self.critic2, state, action)

    def store_observe(self, transition):
        """
        Add a transition sample to the replay buffer. Transition samples will be used in update()
        observe() is used during training.

        Args:
            transition: A transition object. Could be tuple or list
        """
        self.rpb.append(transition)

    def set_reward_func(self, rf):
        """
        Set reward function, default reward function is bellman function with no extra inputs
        """
        self.reward_func = rf

    def set_action_transform_func(self, tf):
        """
        Set action transform function. The transform function is used to transform the output
        of actor to the input of critic
        """
        self.action_trans_func = tf

    def set_update_rate(self, rate=0.01):
        self.update_rate = rate

    def get_replay_buffer(self):
        return self.rpb

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
            next_action = self.action_trans_func(self.act(next_state, True), next_state)
            next_value = self.criticize(next_state, next_action, True)
            next_value2 = self.criticize2(next_state, next_action, True)
            next_value = torch.min(next_value, next_value2)
            reward = reward.view(batch_size, -1)
            terminal = terminal.view(batch_size, -1)
            next_value = next_value.view(batch_size, -1)
            y_i = self.reward_func(reward, self.discount, next_value, terminal, others)

        cur_value = self.criticize(state, action)
        cur_value2 = self.criticize2(state, action)
        value_loss = self.criterion(cur_value, y_i)
        value_loss2 = self.criterion(cur_value2, y_i)

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()
            self.critic2.zero_grad()
            value_loss2.backward()
            self.critic2_optim.step()

        # Update actor network
        cur_action = {"action": self.act(state)}
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

    def update_lr_scheduler(self):
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()

    def load(self, model_dir, network_map, version=-1):
        super(DDPG_TD3, self).load(model_dir, network_map, version)
        with torch.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic2, self.critic2_target)

    def save(self, model_dir, network_map, version=0):
        super(DDPG_TD3, self).save(model_dir, network_map, version)

    @staticmethod
    def bellman_function(reward, discount, next_value, terminal, *_):
        return reward + discount * (1 - terminal) * next_value

    @staticmethod
    def action_trans_func(actions, *args):
        noise = torch.zeros_like(actions)
        noise = noise.data.normal_(0, 0.2)
        noise = torch.clamp(noise, -0.5, 0.5)
        actions = actions + noise
        actions = torch.clamp(actions, min=-1, max=1)
        return { "action": actions }