from typing import Union, Dict, List, Tuple, Callable, Any
import torch as t
import torch.nn as nn
import numpy as np

from machin.model.nets.base import NeuralNetworkModule
from machin.frame.buffers.buffer import Transition, Buffer
from .base import TorchFramework
from .utils import safe_call


class A2C(TorchFramework):
    """
    A2C framework.
    """

    _is_top = ["actor", "critic"]
    _is_restorable = ["actor", "critic"]

    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 criterion: Callable,
                 *_,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple] = (),
                 lr_scheduler_kwargs: Tuple[Dict, Dict] = (),
                 learning_rate: float = 0.001,
                 entropy_weight: float = None,
                 value_weight: float = 0.5,
                 gradient_max: float = np.inf,
                 gae_lambda: float = 1.0,
                 discount: float = 0.99,
                 update_times: int = 50,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 visualize: bool = False,
                 **__):
        """
        Important:
            when given a state, and an optional, action actor must
            at least return two values:

            **1. Action**

              For **contiguous environments**, action must be of shape
              ``[batch_size, action_dim]`` and *clamped by action space*.
              For **discreet environments**, action could be of shape
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

            For discreet environments,
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
                        action_log_prob = dist.log_prob(action)
                        action_entropy = dist.entropy()
                        action = action.clamp(-2.0, 2.0)
                        return action.detach(), action_log_prob, action_entropy

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
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            entropy_weight: Weight of entropy in your loss function, a positive
                entropy weight will minimize entropy, while a negative one will
                maximize entropy.
            value_weight: Weight of critic value loss.
            gradient_max: Maximum gradient.
            gae_lambda: :math:`\\lambda` used in generalized advantage
                estimation.
            discount: :math:`\\gamma` used in the bellman function.
            update_times: Number of update iterations per sample period. Buffer
                will be cleared after ``update()``
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            visualize: Whether visualize the network flow in the first pass.
        """
        self.update_times = update_times
        self.discount = discount
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.grad_max = gradient_max
        self.gae_lambda = gae_lambda
        self.visualize = visualize

        self.actor = actor
        self.critic = critic
        self.actor_optim = optimizer(self.actor.parameters(),
                                     lr=learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(),
                                      lr=learning_rate)
        self.replay_buffer = (Buffer(replay_size, replay_device)
                              if replay_buffer is None
                              else replay_buffer)

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(
                self.actor_optim,
                *lr_scheduler_args[0],
                **lr_scheduler_kwargs[0],
            )
            self.critic_lr_sch = lr_scheduler(
                self.critic_optim,
                *lr_scheduler_args[1],
                **lr_scheduler_kwargs[1]
            )

        self.criterion = criterion

        super(A2C, self).__init__()

    def act(self, state: Dict[str, Any], *_, **__):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(self.actor, state)

    def eval_act(self,
                 state: Dict[str, Any],
                 action: Dict[str, Any],
                 *_, **__):
        """
        Use actor network to evaluate the log-likelihood of a given
        action in the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(self.actor, state, action)

    def criticize(self, state: Dict[str, Any], *_, **__):
        """
        Use critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        """
        return safe_call(self.critic, state)

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer.

        Not suggested, since you will have to calculate "value" and "gae"
        by yourself.
        """
        self.replay_buffer.append(transition, required_attrs=(
            "state", "action", "next_state", "reward", "value",
            "gae", "terminal"
        ))

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.

        "value" and "gae" are automatically calculated.
        """
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
                trans["gae"] = (trans["value"] -
                                self.criticize(trans["state"]).item())
        elif self.gae_lambda == 0.0:
            for trans in episode:
                trans["gae"] = (trans["reward"] +
                                self.discount * (1 - trans["terminal"])
                                * self.criticize(trans["next_state"]).item() -
                                self.criticize(trans["state"]).item())
        else:
            last_critic_value = 0
            last_gae = 0
            for trans in reversed(episode):
                critic_value = self.criticize(trans["state"]).item()
                gae_delta = (trans["reward"] +
                             self.discount * last_critic_value -
                             critic_value)
                last_critic_value = critic_value
                last_gae = trans["gae"] = (last_gae * self.discount
                                           * self.gae_lambda +
                                           gae_delta)

        for trans in episode:
            self.replay_buffer.append(trans, required_attrs=(
                "state", "action", "next_state", "reward", "value",
                "gae", "terminal"
            ))

    def update(self,
               update_value=True,
               update_policy=True,
               concatenate_samples=True,
               **__):
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

        sum_act_policy_loss = 0
        sum_value_loss = 0

        # sample a batch
        batch_size, (state, action, reward, next_state,
                     terminal, target_value, advantage, *others) = \
            self.replay_buffer.sample_batch(-1,
                                            sample_method="all",
                                            concatenate=concatenate_samples,
                                            sample_attrs=[
                                                "state", "action", "reward",
                                                "next_state", "terminal",
                                                "value", "gae", "*"],
                                            additional_concat_attrs=[
                                                "value", "gae"
                                            ])

        # normalize target value
        target_value = ((target_value - target_value.mean()) /
                        (target_value.std() + 1e-5))

        for _ in range(self.update_times):
            value = self.criticize(state)

            if self.entropy_weight is not None:
                new_action, new_action_log_prob, new_action_entropy = \
                    self.eval_act(state, action)
            else:
                new_action, new_action_log_prob, *_ = \
                    self.eval_act(state, action)
                new_action_entropy = None

            new_action_log_prob = new_action_log_prob.view(batch_size, 1)

            # calculate policy loss
            act_policy_loss = -(new_action_log_prob *
                                advantage.to(new_action_log_prob.device))

            if new_action_entropy is not None:
                act_policy_loss += (self.entropy_weight *
                                    new_action_entropy.mean())

            act_policy_loss = act_policy_loss.mean()

            value_loss = (self.criterion(target_value.to(value.device), value) *
                          self.value_weight)

            if self.visualize:
                self.visualize_model(act_policy_loss, "actor")

            # Update actor network
            if update_policy:
                self.actor.zero_grad()
                act_policy_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.grad_max
                )
                self.actor_optim.step()
                sum_act_policy_loss += act_policy_loss.item()

            if self.visualize:
                self.visualize_model(value_loss, "critic")

            # Update critic network
            if update_value:
                self.critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.grad_max
                )
                self.critic_optim.step()
                sum_value_loss += value_loss.item()

        self.replay_buffer.clear()
        return (-sum_act_policy_loss / self.update_times,
                sum_value_loss / self.update_times)

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()
