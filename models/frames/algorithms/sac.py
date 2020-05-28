import numpy as np
import torch
import torch.nn as nn

from models.frames.buffers.replay_buffer import Transition, ReplayBuffer
from models.nets.base import NeuralNetworkModule
from models.noise.action_space_noise import *
from typing import Union, Dict, List
from .base import TorchFramework
from .utils import safe_call, hard_update, soft_update

from utils.visualize import visualize_graph


class SAC(TorchFramework):
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 critic2: Union[NeuralNetworkModule, nn.Module],
                 critic2_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 target_entropy=None,
                 initial_entropy_alpha=1.0,
                 value_weight=0.5,
                 gradient_max=np.inf,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=100,
                 update_rate=0.005,
                 discount=0.99,
                 replay_size=500000,
                 replay_device="cpu",
                 reward_func=None,
                 action_trans_func=None):
        """
        Initialize SAC framework.
        """
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.rpb = ReplayBuffer(replay_size, replay_device)

        self.value_weight = value_weight
        self.entropy_alpha = t.tensor(initial_entropy_alpha, requires_grad=True).view(1)
        self.grad_max = gradient_max
        self.target_entropy = target_entropy

        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.critic2 = critic2
        self.critic2_target = critic2_target
        self.actor_optim = optimizer(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), lr=learning_rate)
        self.critic2_optim = optimizer(self.critic2.parameters(), lr=learning_rate)
        self.alpha_optim = optimizer([self.entropy_alpha], lr=learning_rate)

        # Make sure target and online networks have the same weight
        with torch.no_grad():
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic2, self.critic2_target)

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(self.actor_optim, *lr_scheduler_params[0])
            self.critic_lr_sch = lr_scheduler(self.critic_optim, *lr_scheduler_params[1])
            self.critic2_lr_sch = lr_scheduler(self.critic2_optim, *lr_scheduler_params[1])

        self.criterion = criterion

        self.reward_func = SAC.bellman_function if reward_func is None else reward_func
        self.action_trans_func = SAC.action_transform_function if action_trans_func is None else action_trans_func

        super(SAC, self).__init__()
        self.set_top(["actor", "critic", "critic2", "critic_target", "critic2_target"])
        self.set_restorable(["actor", "critic_target", "critic2_target"])

    def act(self, state):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(self.actor, state)

    def criticize(self, state, action, use_target=False):
        """
        Use critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        """
        if use_target:
            return safe_call(self.critic_target, state, action)
        else:
            return safe_call(self.critic, state, action)

    def criticize2(self, state, action, use_target=False):
        """
        Use critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        """
        if use_target:
            return safe_call(self.critic2_target, state, action)
        else:
            return safe_call(self.critic2, state, action)

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer. Transition samples will be used in update()
        observe() is used during training.
        """
        self.rpb.append(transition,
                        required_attrs=("state", "action", "next_state",
                                       "reward", "terminal"))

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        for trans in episode:
            self.rpb.append(trans,
                            required_attrs=("state", "action", "next_state",
                                           "reward", "terminal"))

    def get_replay_buffer(self):
        return self.rpb

    def update(self, update_value=True, update_policy=True, update_targets=True,
               update_entropy_alpha=True, concatenate_samples=True):
        batch_size, (state, action, reward, next_state, terminal, *others) = \
            self.rpb.sample_batch(self.batch_size, concatenate_samples,
                                  sample_attrs=["state", "action", "reward", "next_state", "terminal", "*"])

        # Update critic network first
        # Generate value reference :math: `y_i` using target actor and target critic
        with torch.no_grad():
            next_action, next_action_log_prob, _,  = self.actor(next_state)
            next_action = self.action_trans_func(next_action, next_state, *others)
            next_value = self.criticize(next_state, next_action, True)
            next_value2 = self.criticize2(next_state, next_action, True)
            next_value = torch.min(next_value, next_value2)
            next_value = next_value.view(batch_size, -1) - \
                         self.entropy_alpha * next_action_log_prob.view(batch_size, -1)
            y_i = self.reward_func(reward, self.discount, next_value, terminal, *others)

        cur_value = self.criticize(state, action)
        cur_value2 = self.criticize2(state, action)
        value_loss = self.criterion(cur_value, y_i.to(cur_value.device))
        value_loss2 = self.criterion(cur_value2, y_i.to(cur_value.device))

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

            self.critic2.zero_grad()
            value_loss2.backward()
            self.critic2_optim.step()

        # Update actor network
        cur_action, cur_action_log_prob, *_ = self.actor(next_state)
        cur_action = self.action_trans_func(cur_action, state, *others)
        act_value = self.criticize(state, cur_action)
        act_value2 = self.criticize2(state, cur_action)
        act_value = t.min(act_value, act_value2)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = ((self.entropy_alpha * cur_action_log_prob) - act_value).mean()

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            self.actor_optim.step()

        # Update target networks
        if update_targets:
            soft_update(self.critic_target, self.critic, self.update_rate)

        if update_entropy_alpha:
            alpha_loss = -(t.log(self.entropy_alpha) *
                           (cur_action_log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        # use .item() to prevent memory leakage
        return -act_policy_loss.item(), value_loss.item()

    def update_lr_scheduler(self):
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()

    def load(self, model_dir, network_map=None, version=-1):
        super(SAC, self).load(model_dir, network_map, version)
        with torch.no_grad():
            hard_update(self.critic, self.critic_target)

    @staticmethod
    def action_transform_function(raw_output_action, *_):
        return {"action": raw_output_action}

    @staticmethod
    def bellman_function(reward, discount, next_value, terminal, *_):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * (1 - terminal) * next_value
