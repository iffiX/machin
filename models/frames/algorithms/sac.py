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
                 optimizer,
                 criterion,
                 entropy_weight=None,
                 value_weight=0.5,
                 gradient_max=np.inf,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=100,
                 update_rate=0.005,
                 discount=0.99,
                 replay_size=500000,
                 replay_device="cpu"):
        """
        Initialize SAC framework.
        """
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.rpb = ReplayBuffer(replay_size, replay_device)

        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.grad_max = gradient_max

        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optim = optimizer(self.actor.parameters(), learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), learning_rate)

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(self.actor_optim, *lr_scheduler_params[0])
            self.critic_lr_sch = lr_scheduler(self.critic_optim, *lr_scheduler_params[1])

        self.criterion = criterion

        super(SAC, self).__init__()
        self.set_top(["actor", "critic", "critic_target"])
        self.set_restorable(["actor", "critic_target"])

    def act(self, state):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(self.actor, state)

    def eval_act(self, state, action):
        """
        Use actor network to evaluate the log-likelihood of a given action in the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(self.actor, state, action)

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

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer. Transition samples will be used in update()
        observe() is used during training.
        """
        self.rpb.append(transition,
                        required_keys=("state", "action", "next_state",
                                       "reward", "terminal"))

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        for trans in episode:
            self.rpb.append(trans)

    def get_replay_buffer(self):
        return self.rpb

    def update(self, update_value=True, update_policy=True, concatenate_samples=True):
        sum_act_policy_loss = 0
        sum_value_loss = 0

        # sample a batch
        batch_size, (state, action, reward, next_state,
                     terminal, target_value, advantage, *others) = \
            self.rpb.sample_batch(-1,
                                  sample_method="all",
                                  concatenate=concatenate_samples,
                                  sample_keys=["state", "action", "reward", "next_state",
                                               "terminal", "value", "gae", "*"],
                                  additional_concat_keys=["value", "gae"])

        # normalize target value
        target_value = (target_value - target_value.mean()) / (target_value.std() + 1e-5)

        for i in range(self.update_times):
            value = self.criticize(state)

            if self.entropy_weight is not None:
                new_action, new_action_log_prob, new_action_entropy = self.eval_act(state, action)

            else:
                new_action, new_action_log_prob, *_ = self.eval_act(state, action)

            new_action_log_prob = new_action_log_prob.view(batch_size, 1)

            # calculate policy loss
            act_policy_loss = -new_action_log_prob * advantage

            if self.entropy_weight is not None:
                act_policy_loss += self.entropy_weight * new_action_entropy.mean()

            act_policy_loss = act_policy_loss.mean()

            value_loss = self.criterion(target_value.to(value.device), value) * self.value_weight

            # Update actor network
            if update_policy:
                self.actor.zero_grad()
                act_policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_max)
                self.actor_optim.step()
                sum_act_policy_loss += act_policy_loss.item()

            # Update critic network
            if update_value:
                self.critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_max)
                self.critic_optim.step()
                sum_value_loss += value_loss.item()

        self.rpb.clear()
        return -sum_act_policy_loss / self.update_times, sum_value_loss / self.update_times

    def update_lr_scheduler(self):
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()

    def load(self, model_dir, network_map=None, version=-1):
        super(SAC, self).load(model_dir, network_map, version)
        with torch.no_grad():
            hard_update(self.critic, self.critic_target)