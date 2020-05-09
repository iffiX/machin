import torch
import torch.nn as nn
from typing import Union, Dict

from .base import TorchFramework
from .utils import hard_update, soft_update, safe_call
from .replay_buffer import Transition, ReplayBuffer

from ..models.base import NeuralNetworkModule
from ..noise.action_space_noise import *

# in case you need to debug your network in ddpg
from utils.visualize import visualize_graph


class DDPG(TorchFramework):
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 actor_target: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
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
        Initialize DDPG framework.
        """
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.rpb = ReplayBuffer(replay_size, replay_device)

        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optim = optimizer(self.actor.parameters(), learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), learning_rate)

        # Make sure target and online networks have the same weight
        with torch.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(self.actor_optim, *lr_scheduler_params[0])
            self.critic_lr_sch = lr_scheduler(self.critic_optim, *lr_scheduler_params[1])

        self.criterion = criterion

        self.reward_func = DDPG.bellman_function if reward_func is None else reward_func
        self.action_trans_func = DDPG.action_transform_function if action_trans_func is None else action_trans_func

        super(DDPG, self).__init__()
        self.set_top(["actor", "critic", "actor_target", "critic_target"])
        self.set_restorable(["actor_target", "critic_target"])

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
        Use actor network to give a policy (with noise added) to the current state.

        Returns:
            Policy (with noise) produced by actor.
        """
        if mode == "uniform":
            return add_uniform_noise_to_action(self.act(state, use_target), noise_param, ratio)
        elif mode == "normal":
            return add_normal_noise_to_action(self.act(state, use_target), noise_param, ratio)
        else:
            raise RuntimeError("Unknown noise type: " + str(mode))

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

    def store_observe(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer. Transition samples will be used in update()
        observe() is used during training.
        """
        self.rpb.append(transition)

    def set_reward_func(self, rf):
        """
        Set reward function, default reward function is bellman function with no extra inputs
        """
        self.reward_func = rf

    def set_action_transform_function(self, tf):
        """
        Set action transform function. The transform function is used to transform the output
        of actor to the input of critic

        Action transform function must accept:
            1. raw action from the actor model
            2. concatenated next_state dictionary from the transition object
            3. any other concatenated lists of custom keys from the transition object
        and returns:
            1. a dictionary similar to action dictinary from the transition object
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
            self.rpb.sample_batch(self.batch_size, concatenate_samples,
                                  sample_keys=["state", "action", "reward", "next_state", "terminal", "*"])

        # Update critic network first
        # Generate value reference :math: `y_i` using target actor and target critic
        with torch.no_grad():
            next_action = self.action_trans_func(self.act(next_state, True), next_state, *others)
            next_value = self.criticize(next_state, next_action, True)
            next_value = next_value.view(batch_size, -1)
            y_i = self.reward_func(reward, self.discount, next_value, terminal, *others)

        cur_value = self.criticize(state, action)
        value_loss = self.criterion(cur_value, y_i.to(cur_value.device))

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

        # Update actor network
        cur_action = self.action_trans_func(self.act(state), state, *others)
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

        # use .item() to prevent memory leakage
        return -act_policy_loss.item(), value_loss.item()

    def update_lr_scheduler(self):
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()

    def load(self, model_dir, network_map=None, version=-1):
        super(DDPG, self).load(model_dir, network_map, version)
        with torch.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)

    def save(self, model_dir, network_map=None, version=0):
        super(DDPG, self).save(model_dir, network_map, version)

    @staticmethod
    def action_transform_function(raw_output_action, *_):
        return {"action": raw_output_action}

    @staticmethod
    def bellman_function(reward, discount, next_value, terminal, *_):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * (1 - terminal) * next_value
