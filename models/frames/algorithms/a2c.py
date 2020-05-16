import numpy as np
import torch
import torch.nn as nn

from models.frames.buffers.replay_buffer import Transition, ReplayBuffer
from models.nets.base import NeuralNetworkModule
from models.noise.action_space_noise import *
from typing import Union, Dict, List
from .base import TorchFramework
from .utils import safe_call

from utils.visualize import visualize_graph


class A2C(TorchFramework):
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 entropy_weight=None,
                 value_weight=0.5,
                 gradient_max=np.inf,
                 gae_lambda=1.0,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 update_times=50,
                 discount=0.99,
                 replay_size=5000,
                 replay_device="cpu"):
        """
        Initialize A2C framework.
        Note: when given a state, (and an optional action) actor must at least return two
        values:
        1. Action
            For contiguous environments, action must be of shape [batch_size, action_dim]
            and clamped to environment limits.
            For discreet environments, action must be of shape [batch_size, action_dim],
            it could be a categorical encoded integer, or a one hot vector.

            Actions are given by samples during training in PPO framework. When actor is
            given a batch of actions and states, it must evaluate the states, and return
            the log likelihood of the given actions instead of re-sampling actions.

        2. Log likelihood of action (action probability)
            For contiguous environments, action's are not directly output by your actor,
            otherwise it would be rather inconvenient to generate this value, instead, your
            actor network should output parameters for a certain distribution (eg: normal)
            and then be drawn from it.

            For discreet environments, action probability is the one of the values in your
            one-hot vector. It is recommended to sample from torch.distribution.Categorical,
            instead of sampling by yourself.

            Action probability must be differentiable, actor will receive its gradient from
            the gradient of action probability.

        The third entropy value is optional:
        3. Entropy of action distribution (Optional)
            Entropy is usually calculated using dist.entropy(), it will be considered if you
            have specified the entropy_weight argument.

            An example of your actor in contiguous environments::

                class ActorNet(nn.Module):
                    def __init__(self):
                        super(ActorNet, self).__init__()
                        self.fc = nn.Linear(3, 100)
                        self.mu_head = nn.Linear(100, 1)
                        self.sigma_head = nn.Linear(100, 1)

                    def forward(self, state, action=None):
                        x = t.relu(self.fc(state))
                        mu = 2.0 * t.tanh(self.mu_head(x))
                        sigma = F.softplus(self.sigma_head(x))
                        dist = Normal(mu, sigma)
                        action = action if action is not None else dist.sample()
                        action_log_prob = dist.log_prob(action)
                        action_entropy = dist.entropy()
                        action = action.clamp(-2.0, 2.0)
                        return action.detach(), action_log_prob, action_entropy

        """
        self.update_times = update_times
        self.discount = discount
        self.rpb = ReplayBuffer(replay_size, replay_device)

        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.surr_clip = surrogate_loss_clip
        self.grad_max = gradient_max
        self.gae_lambda = gae_lambda

        self.actor = actor
        self.critic = critic
        self.actor_optim = optimizer(self.actor.parameters(), learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), learning_rate)

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(self.actor_optim, *lr_scheduler_params[0])
            self.critic_lr_sch = lr_scheduler(self.critic_optim, *lr_scheduler_params[1])

        self.criterion = criterion

        super(A2C, self).__init__()
        self.set_top(["actor", "critic"])
        self.set_restorable(["actor", "critic"])

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

    def criticize(self, state):
        """
        Use critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        """
        return safe_call(self.critic, state)

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer. Transition samples will be used in update()
        observe() is used during training.
        """
        self.rpb.append(transition,
                        required_keys=("state", "action", "next_state",
                                       "reward", "value", "gae", "terminal"))

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        if not isinstance(episode[0], Transition):
            episode = [Transition(**trans) for trans in episode]

        episode[-1]["value"] = episode[-1].reward

        # calculate value for each transition
        for i in reversed(range(1, len(episode))):
            episode[i - 1]["value"] = \
                episode[i]["value"] * self.discount + episode[i - 1]["reward"]

        # calculate advantage
        if self.gae_lambda == 1.0:
            for trans in episode:
                trans["gae"] = trans["value"] - self.criticize(trans["state"]).item()
        elif self.gae_lambda == 0.0:
            for trans in episode:
                trans["gae"] = trans["reward"] + \
                               self.discount * (1 - trans["terminal"]) * \
                               self.criticize(trans["next_state"]).item() \
                               - self.criticize(trans["state"]).item()
        else:
            last_critic_value = 0
            last_gae = 0
            for trans in reversed(episode):
                critic_value = self.criticize(trans["state"]).item()
                gae_delta = trans["reward"] + self.discount * last_critic_value - critic_value
                last_critic_value = critic_value
                last_gae = trans["gae"] = last_gae * self.discount * self.gae_lambda + gae_delta

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
