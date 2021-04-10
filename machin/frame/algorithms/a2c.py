from typing import Union, Dict, List, Tuple, Callable, Any
from copy import deepcopy
import torch as t
import torch.nn as nn
import numpy as np

from machin.model.nets.base import NeuralNetworkModule
from machin.frame.buffers.buffer import Transition, Buffer
from .base import TorchFramework, Config
from .utils import (
    safe_call,
    assert_and_get_valid_models,
    assert_and_get_valid_optimizer,
    assert_and_get_valid_criterion,
    assert_and_get_valid_lr_scheduler,
)


class A2C(TorchFramework):
    """
    A2C framework.
    """

    _is_top = ["actor", "critic"]
    _is_restorable = ["actor", "critic"]

    def __init__(
        self,
        actor: Union[NeuralNetworkModule, nn.Module],
        critic: Union[NeuralNetworkModule, nn.Module],
        optimizer: Callable,
        criterion: Callable,
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple, Tuple] = None,
        lr_scheduler_kwargs: Tuple[Dict, Dict] = None,
        batch_size: int = 100,
        actor_update_times: int = 5,
        critic_update_times: int = 10,
        actor_learning_rate: float = 0.001,
        critic_learning_rate: float = 0.001,
        entropy_weight: float = None,
        value_weight: float = 0.5,
        gradient_max: float = np.inf,
        gae_lambda: float = 1.0,
        discount: float = 0.99,
        normalize_advantage: bool = True,
        replay_size: int = 500000,
        replay_device: Union[str, t.device] = "cpu",
        replay_buffer: Buffer = None,
        visualize: bool = False,
        visualize_dir: str = "",
        **__
    ):
        """
        Important:
            when given a state, and an optional, action actor must
            at least return two values:

            **1. Action**

              For **contiguous environments**, action must be of shape
              ``[batch_size, action_dim]`` and *clamped by action space*.
              For **discrete environments**, action could be of shape
              ``[batch_size, action_dim]`` if it is a one hot vector, or
              ``[batch_size, 1]`` if it is a categorically encoded integer.

            **2. Log likelihood of action (action probability)**

              For either type of environment, log likelihood is of shape
              ``[batch_size, 1]``.

              Action probability must be differentiable, Gradient of actor
              is calculated from the gradient of action probability.

            The third entropy value is optional:

            **3. Entropy of action distribution**

              Entropy is usually calculated using dist.entropy(), its shape
              is ``[batch_size, 1]``. You must specify ``entropy_weight``
              to make it effective.

        Hint:
            For contiguous environments, action's are not directly output by
            your actor, otherwise it would be rather inconvenient to calculate
            the log probability of action. Instead, your actor network should
            output parameters for a certain distribution
            (eg: :class:`~torch.distributions.categorical.Normal`)
            and then draw action from it.

            For discrete environments,
            :class:`~torch.distributions.categorical.Categorical` is sufficient,
            since differentiable ``rsample()`` is not needed.

            This trick is also known as **reparameterization**.

        Hint:
            Actions are from samples during training in the actor critic
            family (A2C, A3C, PPO, TRPO, IMPALA).

            When your actor model is given a batch of actions and states, it
            must evaluate the states, and return the log likelihood of the
            given actions instead of re-sampling actions.

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
                        action = (action
                                  if action is not None
                                  else dist.sample())
                        action_entropy = dist.entropy()
                        action = action.clamp(-2.0, 2.0)
                        action_log_prob = dist.log_prob(action)
                        return action, action_log_prob, action_entropy

        Hint:
            Entropy weight is usually negative, to increase exploration.

            Value weight is usually 0.5. So critic network converges less
            slowly than the actor network and learns more conditions.

            Update equation is equivalent to:

            :math:`Loss= w_e * Entropy + w_v * Loss_v + w_a * Loss_a`
            :math:`Loss_a = -log\\_likelihood * advantage`
            :math:`Loss_v = criterion(target\\_bellman\\_value - V(s))`

        Args:
            actor: Actor network module.
            critic: Critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used during training.
            actor_update_times: Times to update actor in ``update()``.
            critic_update_times: Times to update critic in ``update()``.
            actor_learning_rate: Learning rate of the actor optimizer,
                not compatible with ``lr_scheduler``.
            critic_learning_rate: Learning rate of the critic optimizer,
                not compatible with ``lr_scheduler``.
            entropy_weight: Weight of entropy in your loss function, a positive
                entropy weight will minimize entropy, while a negative one will
                maximize entropy.
            value_weight: Weight of critic value loss.
            gradient_max: Maximum gradient.
            gae_lambda: :math:`\\lambda` used in generalized advantage
                estimation.
            discount: :math:`\\gamma` used in the bellman function.
            normalize_advantage: Whether to normalize the advantage function.
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            visualize: Whether visualize the network flow in the first pass.
            visualize_dir: Visualized graph save directory.
        """
        self.batch_size = batch_size
        self.actor_update_times = actor_update_times
        self.critic_update_times = critic_update_times
        self.discount = discount
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.gradient_max = gradient_max
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage
        self.visualize = visualize
        self.visualize_dir = visualize_dir

        self.actor = actor
        self.critic = critic
        self.actor_optim = optimizer(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), lr=critic_learning_rate)
        self.replay_buffer = (
            Buffer(replay_size, replay_device)
            if replay_buffer is None
            else replay_buffer
        )

        if lr_scheduler is not None:
            if lr_scheduler_args is None:
                lr_scheduler_args = ((), ())
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = ({}, {})
            self.actor_lr_sch = lr_scheduler(
                self.actor_optim, *lr_scheduler_args[0], **lr_scheduler_kwargs[0],
            )
            self.critic_lr_sch = lr_scheduler(
                self.critic_optim, *lr_scheduler_args[1], **lr_scheduler_kwargs[1]
            )

        self.criterion = criterion

        super().__init__()

    @property
    def optimizers(self):
        return [self.actor_optim, self.critic_optim]

    @optimizers.setter
    def optimizers(self, optimizers):
        self.actor_optim, self.critic_optim = optimizers

    @property
    def lr_schedulers(self):
        if hasattr(self, "actor_lr_sch") and hasattr(self, "critic_lr_sch"):
            return [self.actor_lr_sch, self.critic_lr_sch]
        return []

    def act(self, state: Dict[str, Any], *_, **__):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Anything produced by actor.
        """
        # No need to safe_return because the number of
        # returned values is always more than one
        return safe_call(self.actor, state)

    def _eval_act(self, state: Dict[str, Any], action: Dict[str, Any], *_, **__):
        """
        Use actor network to evaluate the log-likelihood of a given
        action in the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(self.actor, state, action)

    def _criticize(self, state: Dict[str, Any], *_, **__):
        """
        Use critic network to evaluate current value.

        Returns:
            Value of shape ``[batch_size, 1]``
        """
        return safe_call(self.critic, state)[0]

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer.

        Not suggested, since you will have to calculate "value" and "gae"
        by yourself.
        """
        self.replay_buffer.append(
            transition,
            required_attrs=(
                "state",
                "action",
                "next_state",
                "reward",
                "value",
                "gae",
                "terminal",
            ),
        )

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.

        "value" and "gae" are automatically calculated.
        """
        episode[-1]["value"] = episode[-1]["reward"]

        # calculate value for each transition
        for i in reversed(range(1, len(episode))):
            episode[i - 1]["value"] = (
                episode[i]["value"] * self.discount + episode[i - 1]["reward"]
            )

        # calculate advantage
        if self.gae_lambda == 1.0:
            for trans in episode:
                trans["gae"] = trans["value"] - self._criticize(trans["state"]).item()
        elif self.gae_lambda == 0.0:
            for trans in episode:
                trans["gae"] = (
                    trans["reward"]
                    + self.discount
                    * (1 - float(trans["terminal"]))
                    * self._criticize(trans["next_state"]).item()
                    - self._criticize(trans["state"]).item()
                )
        else:
            last_critic_value = 0
            last_gae = 0
            for trans in reversed(episode):
                critic_value = self._criticize(trans["state"]).item()
                gae_delta = (
                    trans["reward"]
                    + self.discount * last_critic_value * (1 - float(trans["terminal"]))
                    - critic_value
                )
                last_critic_value = critic_value
                last_gae = trans["gae"] = (
                    last_gae
                    * self.discount
                    * (1 - float(trans["terminal"]))
                    * self.gae_lambda
                    + gae_delta
                )

        for trans in episode:
            self.replay_buffer.append(
                trans,
                required_attrs=(
                    "state",
                    "action",
                    "next_state",
                    "reward",
                    "value",
                    "gae",
                    "terminal",
                ),
            )

    def update(
        self, update_value=True, update_policy=True, concatenate_samples=True, **__
    ):
        """
        Update network weights by sampling from buffer. Buffer
        will be cleared after update is finished.

        Args:
            update_value: Whether update the Q network.
            update_policy: Whether update the actor network.
            concatenate_samples: Whether concatenate the samples.

        Returns:
            mean value of estimated policy value, value loss
        """
        sum_act_loss = 0
        sum_value_loss = 0
        self.actor.train()
        self.critic.train()
        for _ in range(self.actor_update_times):
            # sample a batch
            batch_size, (state, action, advantage) = self.replay_buffer.sample_batch(
                self.batch_size,
                sample_method="random_unique",
                concatenate=concatenate_samples,
                sample_attrs=["state", "action", "gae"],
                additional_concat_attrs=["gae"],
            )

            # normalize advantage
            if self.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

            if self.entropy_weight is not None:
                __, action_log_prob, new_action_entropy, *_ = self._eval_act(
                    state, action
                )
            else:
                __, action_log_prob, *_ = self._eval_act(state, action)
                new_action_entropy = None

            action_log_prob = action_log_prob.view(batch_size, 1)

            # calculate policy loss
            act_policy_loss = -(action_log_prob * advantage.type_as(action_log_prob))

            if new_action_entropy is not None:
                act_policy_loss += self.entropy_weight * new_action_entropy.mean()

            act_policy_loss = act_policy_loss.mean()
            sum_act_loss += act_policy_loss.item()

            if self.visualize:
                self.visualize_model(act_policy_loss, "actor", self.visualize_dir)

            # Update actor network
            if update_policy:
                self.actor.zero_grad()
                self._backward(act_policy_loss)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_max)
                self.actor_optim.step()

        for _ in range(self.critic_update_times):
            # sample a batch
            batch_size, (state, target_value) = self.replay_buffer.sample_batch(
                self.batch_size,
                sample_method="random_unique",
                concatenate=concatenate_samples,
                sample_attrs=["state", "value"],
                additional_concat_attrs=["value"],
            )
            # calculate value loss
            value = self._criticize(state)
            value_loss = (
                self.criterion(target_value.type_as(value), value) * self.value_weight
            )
            sum_value_loss += value_loss.item()

            if self.visualize:
                self.visualize_model(value_loss, "critic", self.visualize_dir)

            # Update critic network
            if update_value:
                self.critic.zero_grad()
                self._backward(value_loss)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_max)
                self.critic_optim.step()

        self.replay_buffer.clear()
        self.actor.eval()
        self.critic.eval()
        return (
            -sum_act_loss / self.actor_update_times,
            sum_value_loss / self.critic_update_times,
        )

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        default_values = {
            "models": ["Actor", "Critic"],
            "model_args": ((), ()),
            "model_kwargs": ({}, {}),
            "optimizer": "Adam",
            "criterion": "MSELoss",
            "criterion_args": (),
            "criterion_kwargs": {},
            "lr_scheduler": None,
            "lr_scheduler_args": None,
            "lr_scheduler_kwargs": None,
            "batch_size": 100,
            "actor_update_times": 5,
            "critic_update_times": 10,
            "actor_learning_rate": 0.001,
            "critic_learning_rate": 0.001,
            "entropy_weight": None,
            "value_weight": 0.5,
            "gradient_max": np.inf,
            "gae_lambda": 1.0,
            "discount": 0.99,
            "normalize_advantage": True,
            "replay_size": 500000,
            "replay_device": "cpu",
            "replay_buffer": None,
            "visualize": False,
            "visualize_dir": "",
        }
        config = deepcopy(config)
        config["frame"] = "A2C"
        if "frame_config" not in config:
            config["frame_config"] = default_values
        else:
            config["frame_config"] = {**config["frame_config"], **default_values}
        return config

    @classmethod
    def init_from_config(
        cls,
        config: Union[Dict[str, Any], Config],
        model_device: Union[str, t.device] = "cpu",
    ):
        f_config = deepcopy(config["frame_config"])
        models = assert_and_get_valid_models(f_config["models"])
        model_args = f_config["model_args"]
        model_kwargs = f_config["model_kwargs"]
        models = [
            m(*arg, **kwarg).to(model_device)
            for m, arg, kwarg in zip(models, model_args, model_kwargs)
        ]
        optimizer = assert_and_get_valid_optimizer(f_config["optimizer"])
        criterion = assert_and_get_valid_criterion(f_config["criterion"])(
            *f_config["criterion_args"], **f_config["criterion_kwargs"]
        )
        lr_scheduler = f_config["lr_scheduler"] and assert_and_get_valid_lr_scheduler(
            f_config["lr_scheduler"]
        )

        f_config["optimizer"] = optimizer
        f_config["criterion"] = criterion
        f_config["lr_scheduler"] = lr_scheduler
        frame = cls(*models, **f_config)
        return frame
