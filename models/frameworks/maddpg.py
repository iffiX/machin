import time
import random
import inspect
import copy
import numpy as np
import torch
import torch.nn as nn

from .base import TorchFramework
from .ddpg import soft_update, hard_update, safe_call, ReplayBuffer
from typing import Union, List, Tuple

from utils.visualize import visualize_graph


class MADDPG_ReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size):
        super(MADDPG_ReplayBuffer, self).__init__(buffer_size)

    def sample_batch(self, batch_size, concatenate=True, device=None, sample_keys=None):
        """
        Sample a random batch from replay buffer.

        Args:
            batch_size: Maximum size of the sample.
            device:     Device to copy to.
            sample_keys: If sample_keys is specified, then only specified keys
                         of the transition object will be sampled and stacked.

        Returns:
            None if no batch is sampled.
            Or a tuple of sampled results, the tensors in "state", "action" and
            "next_state" dictionaries will be concatenated in dimension 0. Values
            of "reward" and other custom keys will be concatenated together as tuples.
        """
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
            real_num = self.size()
        else:
            batch = random.sample(self.buffer, batch_size)
            real_num = batch_size

        if len(batch) == 0:
            return 0, ()

        if sample_keys is None:
            sample_keys = batch[0].keys()
        result = []
        for k in sample_keys:
            if k in ("state", "action", "other_actions", "next_state"):
                tmp_dict = {}
                for sub_k in batch[0][k].keys():
                    if concatenate:
                        tmp_dict[sub_k] = torch.cat([item[k][sub_k] for item in batch], dim=0) \
                                               .to(device)
                    else:
                        if batch_size == 1:
                            tmp_dict[sub_k] = batch[0][k][sub_k]
                        else:
                            tmp_dict[sub_k] = [item[k][sub_k] for item in batch]
                result.append(tmp_dict)
            elif k == "reward":
                if torch.is_tensor(batch[0][k]) and len(batch[0][k].shape) > 0:
                    result.append(torch.cat([item[k] for item in batch], dim=0).to(device).view(real_num, -1))
                else:
                    result.append(torch.tensor([float(item[k]) for item in batch], device=device).view(real_num, -1))
            elif k == "terminal":
                result.append(torch.tensor([float(item[k]) for item in batch], device=device).view(real_num, -1))
            elif k == "*":
                # select custom keys
                for remain_k in batch[0].keys():
                    if remain_k not in ("state", "action", "other_actions", "next_state", "reward", "terminal"):
                        result.append(tuple(item[remain_k] for item in batch))
            else:
                result.append(tuple(item[k] for item in batch))
        return real_num, tuple(result)




class MADDPG(TorchFramework):
    # TODO: multi-threading &|| multi processing
    def __init__(self,
                 actors: List[nn.Module],
                 actor_targets: List[nn.Module],
                 critics: List[nn.Module],
                 critic_targets: List[nn.Module],
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
                 action_trans_func=None,
                 action_ccat_func=None):
        """
        Initialize DDPG framework.
        """
        self.device = device
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.rpb = MADDPG_ReplayBuffer(replay_size)

        self.actors = actors
        self.actor_targets = actor_targets
        self.critics = critics
        self.critic_targets = critic_targets
        self.actor_optims = [optimizer(ac.parameters(), lr=learning_rate) for ac in actors]
        self.critic_optims = [optimizer(cr.parameters(), lr=learning_rate) for cr in critics]

        # result parameters will be averaged before saving
        self.actor_target = copy.deepcopy(actor_targets[0])
        self.critic_target = copy.deepcopy(critic_targets[0])

        # Make sure target and online networks have the same weight
        with torch.no_grad():
            for actor, actor_target in zip(actors, actor_targets):
                hard_update(actor, actor_target)
            for critic, critic_target in zip(critics, critic_targets):
                hard_update(critic, critic_target)

        if lr_scheduler is not None:
            self.actor_lr_schs = [lr_scheduler(ac_opt, *lr_scheduler_params[0]) for ac_opt in self.actor_optims]
            self.critic_lr_schs = [lr_scheduler(cr_opt, *lr_scheduler_params[1]) for cr_opt in self.critic_optims]

        self.criterion = criterion

        self.reward_func = MADDPG.bellman_function if reward_func is None else reward_func
        self.action_trans_func = MADDPG.action_trans_func if action_trans_func is None else action_trans_func
        self.action_ccat_func = MADDPG.action_ccat_func if action_ccat_func is None else action_ccat_func

        super(MADDPG, self).__init__()
        self.set_top([])
        self.set_restorable(["actor_target", "critic_target"])

    def act(self, index, state, use_target=False):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Policy produced by actor.
        """
        if use_target:
            return safe_call(self.actor_targets[index], state)
        else:
            return safe_call(self.actors[index], state)

    def act_with_noise(self, index, state, noise_param=(0.0, 1.0),
                       ratio=1.0, mode="uniform", use_target=False):
        """
        Use actor network to give a policy (with uniform noise added) to the current state.

        Args:
            noise_param: A single tuple or a list of tuples specifying noise params
            for each column (last dimension) of action.
        Returns:
            Policy (with uniform noise) produced by actor.
        """
        if mode == "uniform":
            return self.add_uniform_noise_to_action(self.act(index, state, use_target),
                                                    noise_param, ratio)
        elif mode == "normal":
            return self.add_normal_noise_to_action(self.act(index, state, use_target),
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

    def criticize(self, index, state, action, other_actions, use_target=False):
        """
        Use the first critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        Notes:
            State and action will be concatenated in dimension 1
        """
        if use_target:
            return safe_call(self.critic_targets[index], state, action, other_actions)
        else:
            return safe_call(self.critics[index], state, action, other_actions)

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

    def update(self, update_value=True, update_policy=True, update_targets=True,
               average_target_parametrs=False, concatenate_samples=True):
        """
        Update network weights by sampling from replay buffer.

        Returns:
            (mean value of estimated policy value, value loss)

        Note: currently agents share the same replay buffer
        """
        batch_size, (state, action, other_actions, reward, next_state, terminal, *others) = \
            self.rpb.sample_batch(self.batch_size, concatenate_samples, self.device,
                                  sample_keys=["state", "action", "other_actions",
                                               "reward", "next_state", "terminal", "*"])

        # Update critic network first
        # Generate value reference :math: `y_i` using target actor and target critic
        total_act_policy_loss = 0
        total_value_loss = 0
        for i in range(len(self.actors)):
            others = [k for k in range(len(self.actors)) if k != i]
            with torch.no_grad():
                next_action, next_other_actions = self.action_trans_func(
                    self.act(i, next_state, True),
                    self.action_ccat_func(
                        [self.act(k, next_state, True) for k in others]
                    ),
                    next_state
                )
                next_value = self.criticize(i, next_state, next_action, next_other_actions, True)
                next_value = next_value.view(batch_size, -1)
                y_i = self.reward_func(reward, self.discount, next_value, terminal, others)

            cur_value = self.criticize(i, state, action, other_actions)
            value_loss = self.criterion(cur_value, y_i)
            total_value_loss += value_loss

            if update_value:
                self.critics[i].zero_grad()
                value_loss.backward()
                self.critic_optims[i].step()

            # Update actor network
            # TODO: simplify this line
            cur_action, _ = self.action_trans_func(self.act(i, state), other_actions, state)
            act_value = self.criticize(i, state, cur_action, other_actions)

            # "-" is applied because we want to maximize J_b(u),
            # but optimizer workers by minimizing the target
            act_policy_loss = -act_value.mean()
            total_act_policy_loss += act_policy_loss.detach()

            if update_policy:
                self.actors[i].zero_grad()
                act_policy_loss.backward()
                self.actor_optims[i].step()

            # Update target networks
            if update_targets:
                soft_update(self.actor_targets[i], self.actors[i], self.update_rate)
                soft_update(self.critic_targets[i], self.critics[i], self.update_rate)

        if average_target_parametrs:
            self.average_target_parameters()

        total_act_policy_loss /= len(self.actors)
        total_value_loss /= len(self.actors)
        # use .item() to prevent memory leakage
        return -total_act_policy_loss.item(), total_value_loss.item()

    def update_lr_scheduler(self):
        if hasattr(self, "actor_lr_schs"):
            for actor_lr_sch in self.actor_lr_schs:
                actor_lr_sch.step()
        if hasattr(self, "critic_lr_schs"):
            for critic_lr_sch in self.critic_lr_schs:
                critic_lr_sch.step()

    def load(self, model_dir, network_map, version=-1):
        super(MADDPG, self).load(model_dir, network_map, version)
        with torch.no_grad():
            for actor in self.actors:
                hard_update(actor, self.actor_target)
            for critic in self.critics:
                hard_update(critic, self.critic_target)
            for actor in self.actor_targets:
                hard_update(actor, self.actor_target)
            for critic in self.critic_targets:
                hard_update(critic, self.critic_target)

    def save(self, model_dir, network_map, version=0):
        # average parameters
        with torch.no_grad():
            actor_params = [net.parameters() for net in [self.actor_target] + self.actor_targets]
            for target_param, *params in zip(*actor_params):
                target_param.data.copy_(
                    torch.mean(torch.stack([p.to(target_param.device) for p in params], dim=0), dim=0)
                )
            critic_params = [net.parameters() for net in [self.critic_target] + self.critic_targets]
            for target_param, *params in zip(*critic_params):
                target_param.data.copy_(
                    torch.mean(torch.stack([p.to(target_param.device) for p in params], dim=0), dim=0)
                )
        super(MADDPG, self).save(model_dir, network_map, version)

    def average_target_parameters(self):
        with torch.no_grad():
            actor_params = [net.parameters() for net in [self.actor_target] + self.actor_targets]
            for target_param, *params in zip(*actor_params):
                target_param.data.copy_(
                    torch.mean(torch.stack([p.to(target_param.device) for p in params], dim=0), dim=0)
                )
                for p in params:
                    p.data.copy_(target_param.to(p.device))
            critic_params = [net.parameters() for net in [self.critic_target] + self.critic_targets]
            for target_param, *params in zip(*critic_params):
                target_param.data.copy_(
                    torch.mean(torch.stack([p.to(target_param.device) for p in params], dim=0), dim=0)
                )
                for p in params:
                    p.data.copy_(target_param.to(p.device))

    @staticmethod
    def bellman_function(reward, discount, next_value, terminal, *_):
        return reward + discount * (1 - terminal) * next_value

    @staticmethod
    def action_trans_func(action, other_actions, *_):
        return {"action": action}, {"other_actions": other_actions}

    @staticmethod
    def action_ccat_func(raw_other_actions_list):
        if len(raw_other_actions_list) > 0:
            return torch.cat([act for act in raw_other_actions_list], dim=1)
        else:
            return None


