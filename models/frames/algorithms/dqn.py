import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Union, Dict, List

from models.frames.buffers.buffer import Transition, Buffer
from models.nets.base import NeuralNetworkModule
from models.noise.action_space_noise import *

from .base import TorchFramework
from .utils import hard_update, soft_update, safe_call

from utils.visualize import visualize_graph


class DQN(TorchFramework):
    def __init__(self,
                 qnet: Union[NeuralNetworkModule, nn.Module],
                 qnet_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=100,
                 update_rate=1.0,
                 discount=0.99,
                 replay_size=500000,
                 replay_device="cpu",
                 reward_func=None,
                 mode="double"):
        """
        Initialize DQN framework.
        Note: DQN is only available for discreet environments.
        Note: Vanilla DQN only needs one network, so use the same model (which is used as qnet)
              as the qnet_target.
        Note: In order to implement dueling DQN, you should create two dense output layers
              in your q network:

                    self.fc_adv = nn.Linear(in_features=..., out_features=num_actions)
                    self.fc_val = nn.Linear(in_features=..., out_features=1)

              then in your forward() method, you should implement output as:

                    adv = self.fc_adv(input)
                    val = self.fc_val(input).expand(self.batch_sze, self.num_actions)
                    return val + adv - adv.mean(1, keepdim=True)

        Args:
            mode: one of "vanilla", "fixed_target", "double"
        """
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.rpb = Buffer(replay_size, replay_device)

        if mode not in {"vanilla", "fixed_target", "double"}:
            raise RuntimeError("Unknown DQN mode: {}".format(mode))
        self.mode = mode

        self.qnet = qnet
        if self.mode == "vanilla":
            self.qnet_target = qnet
        else:
            self.qnet_target = qnet_target
        self.qnet_optim = optimizer(self.qnet.parameters(), learning_rate)


        # Make sure target and online networks have the same weight
        with torch.no_grad():
            hard_update(self.qnet, self.qnet_target)

        if lr_scheduler is not None:
            self.qnet_lr_sch = lr_scheduler(self.qnet_optim, *lr_scheduler_params[1])

        self.criterion = criterion

        self.reward_func = DQN.bellman_function if reward_func is None else reward_func

        super(DQN, self).__init__()
        self.set_top(["qnet", "qnet_target"])
        self.set_restorable(["qnet_target"])

    def act_discreet(self, state, use_target=False):
        if use_target:
            result = safe_call(self.qnet_target, state)
        else:
            result = safe_call(self.qnet, state)

        result = t.argmax(result, dim=1).view(-1, 1)
        return result

    def act_discreet_with_noise(self, state, use_target=False):
        if use_target:
            result = safe_call(self.qnet_target, state)
        else:
            result = safe_call(self.qnet, state)

        result = t.softmax(result, dim=1)
        dist = Categorical(result)
        batch_size = result.shape[0]
        return dist.sample([batch_size, 1])

    def criticize(self, state, use_target=False):
        if use_target:
            return safe_call(self.qnet_target, state)
        else:
            return safe_call(self.qnet, state)

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer. Transition samples will be used in update()
        observe() is used during training.
        """
        self.rpb.append(transition)

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        for trans in episode:
            self.rpb.append(trans)

    def set_reward_func(self, rf):
        """
        Set reward function, default reward function is bellman function with no extra inputs
        """
        self.reward_func = rf

    def set_update_rate(self, rate=0.01):
        self.update_rate = rate

    def get_replay_buffer(self):
        return self.rpb

    def update(self, update_value=True, update_targets=True, concatenate_samples=True):
        """
        Update network weights by sampling from replay buffer.

        Returns:
            (mean value of estimated policy value, value loss)
        """
        batch_size, (state, action, reward, next_state, terminal, *others) = \
            self.rpb.sample_batch(self.batch_size, concatenate_samples,
                                  sample_keys=["state", "action", "reward", "next_state", "terminal", "*"])

        if self.mode == "vanilla":
            # target network is the same as the main network
            q_value = self.criticize(state)
            action_value = q_value.gather(dim=1, index=action["action"])
            target_next_q_value = t.max(self.criticize(next_state), dim=1).values.unsqueeze(1)
            y_i = self.reward_func(reward, self.discount, target_next_q_value, terminal, *others)
            value_loss = self.criterion(action_value, y_i.to(action_value.device))

            if update_value:
                self.qnet.zero_grad()
                value_loss.backward()
                self.qnet_optim.step()

        elif self.mode == "fixed_target":
            # Generate value reference :math: `y_i` using target actor and target qnet
            q_value = self.criticize(state)
            action_value = q_value.gather(dim=1, index=action["action"])

            with torch.no_grad():
                target_next_q_value = t.max(self.criticize(next_state, True), dim=1).values.unsqueeze(1)

            y_i = self.reward_func(reward, self.discount, target_next_q_value, terminal, *others)
            value_loss = self.criterion(action_value, y_i.to(action_value.device))

            if update_value:
                self.qnet.zero_grad()
                value_loss.backward()
                self.qnet_optim.step()

            # Update target networks
            if update_targets:
                soft_update(self.qnet_target, self.qnet, self.update_rate)

        elif self.mode == "double":
            # Generate value reference :math: `y_i` using target actor and target qnet
            q_value = self.criticize(state)
            action_value = q_value.gather(dim=1, index=action["action"])

            with torch.no_grad():
                next_q_value = self.criticize(next_state)
                target_next_q_value = self.criticize(next_state, True)
                target_next_q_value = target_next_q_value\
                    .gather(dim=1, index=t.max(next_q_value, dim=1).indicies.unsqueeze(1))

            y_i = self.reward_func(reward, self.discount, target_next_q_value, terminal, *others)
            value_loss = self.criterion(action_value, y_i.to(action_value.device))

            if update_value:
                self.qnet.zero_grad()
                value_loss.backward()
                self.qnet_optim.step()

            # Update target networks
            if update_targets:
                soft_update(self.qnet_target, self.qnet, self.update_rate)

        else:
            raise RuntimeError("Unknown DQN mode: {}".format(self.mode))

        # use .item() to prevent memory leakage
        return value_loss.item()

    def update_lr_scheduler(self):
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "qnet_lr_sch"):
            self.qnet_lr_sch.step()

    def load(self, model_dir, network_map=None, version=-1):
        super(DQN, self).load(model_dir, network_map, version)
        with torch.no_grad():
            hard_update(self.qnet, self.qnet_target)

    def save(self, model_dir, network_map=None, version=0):
        super(DQN, self).save(model_dir, network_map, version)

    @staticmethod
    def bellman_function(reward, discount, next_value, terminal, *_):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * (1 - terminal) * next_value
