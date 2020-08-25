from typing import Union, Dict, List, Tuple, Callable, Any
from torch.distributions import Categorical
import torch as t
import torch.nn as nn
import numpy as np

from machin.frame.buffers.buffer import Transition, Buffer
from machin.frame.noise.action_space_noise import \
    add_normal_noise_to_action, \
    add_clipped_normal_noise_to_action, \
    add_uniform_noise_to_action, \
    add_ou_noise_to_action
from machin.model.nets.base import NeuralNetworkModule
from .base import TorchFramework
from .utils import \
    hard_update, \
    soft_update, \
    safe_call, \
    safe_return, \
    assert_output_is_probs


class DDPG(TorchFramework):
    """
    DDPG framework.
    """

    _is_top = ["actor", "critic", "actor_target", "critic_target"]
    _is_restorable = ["actor_target", "critic_target"]

    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 actor_target: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 criterion: Callable,
                 *_,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple] = None,
                 lr_scheduler_kwargs: Tuple[Dict, Dict] = None,
                 batch_size: int = 100,
                 update_rate: float = 0.001,
                 actor_learning_rate: float = 0.0005,
                 critic_learning_rate: float = 0.001,
                 discount: float = 0.99,
                 gradient_max: float = np.inf,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 visualize: bool = False,
                 visualize_dir: str = "",
                 **__):
        """
        Note:
            Your optimizer will be called as::

                optimizer(network.parameters(), learning_rate)

            Your lr_scheduler will be called as::

                lr_scheduler(
                    optimizer,
                    *lr_scheduler_args[0],
                    **lr_scheduler_kwargs[0],
                )

            Your criterion will be called as::

                criterion(
                    target_value.view(batch_size, 1),
                    predicted_value.view(batch_size, 1)
                )

        Args:
            actor: Actor network module.
            actor_target: Target actor network module.
            critic: Critic network module.
            critic_target: Target critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used during training.
            update_rate: :math:`\\tau` used to update target networks.
                Target parameters are updated as:

                :math:`\\theta_t = \\theta * \\tau + \\theta_t * (1 - \\tau)`
            actor_learning_rate: Learning rate of the actor optimizer,
                not compatible with ``lr_scheduler``.
            critic_learning_rate: Learning rate of the critic optimizer,
                not compatible with ``lr_scheduler``.
            discount: :math:`\\gamma` used in the bellman function.
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            visualize: Whether visualize the network flow in the first pass.
            visualize_dir: Visualized graph save directory.
        """
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.grad_max = gradient_max
        self.visualize = visualize
        self.visualize_dir = visualize_dir

        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optim = optimizer(self.actor.parameters(),
                                     lr=actor_learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(),
                                      lr=critic_learning_rate)
        self.replay_buffer = (Buffer(replay_size, replay_device)
                              if replay_buffer is None
                              else replay_buffer)

        # Make sure target and online networks have the same weight
        with t.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)

        if lr_scheduler is not None:
            if lr_scheduler_args is None:
                lr_scheduler_args = ((), ())
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = ({}, {})
            self.actor_lr_sch = lr_scheduler(
                self.actor_optim,
                *lr_scheduler_args[0],
                **lr_scheduler_kwargs[0]
            )
            self.critic_lr_sch = lr_scheduler(
                self.critic_optim,
                *lr_scheduler_args[1],
                **lr_scheduler_kwargs[1]
            )

        self.criterion = criterion
        super(DDPG, self).__init__()

    def act(self,
            state: Dict[str, Any],
            use_target: bool = False,
            **__):
        """
        Use actor network to produce an action for the current state.

        Args:
            state: Current state.
            use_target: Whether use the target network.

        Returns:
            Any thing returned by your actor network.
        """
        if use_target:
            return safe_return(safe_call(self.actor_target, state))
        else:
            return safe_return(safe_call(self.actor, state))

    def act_with_noise(self,
                       state: Dict[str, Any],
                       noise_param: Any = (0.0, 1.0),
                       ratio: float = 1.0,
                       mode: str = "uniform",
                       use_target: bool = False,
                       **__):
        """
        Use actor network to produce a noisy action for the current state.

        See Also:
             :mod:`machin.frame.noise.action_space_noise`

        Args:
            state: Current state.
            noise_param: Noise params.
            ratio: Noise ratio.
            mode: Noise mode. Supported are:
                ``"uniform", "normal", "clipped_normal", "ou"``
            use_target: Whether use the target network.

        Returns:
            Noisy action of shape ``[batch_size, action_dim]``.
            Any other things returned by your actor network. if they exist.
        """
        if use_target:
            action, *others = safe_call(self.actor_target, state)
        else:
            action, *others = safe_call(self.actor, state)
        if mode == "uniform":
            noisy_action = add_uniform_noise_to_action(
                action, noise_param, ratio
            )
        elif mode == "normal":
            noisy_action = add_normal_noise_to_action(
                action, noise_param, ratio
            )
        elif mode == "clipped_normal":
            noisy_action = add_clipped_normal_noise_to_action(
                action, noise_param, ratio
            )
        elif mode == "ou":
            noisy_action = add_ou_noise_to_action(
                action, noise_param, ratio
            )
        else:
            raise ValueError("Unknown noise type: " + str(mode))

        if len(others) == 0:
            return noisy_action
        else:
            return (noisy_action, *others)

    def act_discrete(self,
                     state: Dict[str, Any],
                     use_target: bool = False,
                     **__):
        """
        Use actor network to produce a discrete action for the current state.

        Notes:
            actor network must output a probability tensor, of shape
            (batch_size, action_dims), and has a sum of 1 for each row
            in dimension 1.

        Args:
            state: Current state.
            use_target: Whether to use the target network.

        Returns:
            Action of shape ``[batch_size, 1]``.
            Action probability tensor of shape ``[batch_size, action_num]``,
            produced by your actor.
            Any other things returned by your Q network. if they exist.
        """
        if use_target:
            action, *others = safe_call(self.actor_target, state)
        else:
            action, *others = safe_call(self.actor, state)

        assert_output_is_probs(action)
        batch_size = action.shape[0]
        result = t.argmax(action, dim=1).view(batch_size, 1)
        return (result, action, *others)

    def act_discrete_with_noise(self,
                                state: Dict[str, Any],
                                use_target: bool = False,
                                **__):
        """
        Use actor network to produce a noisy discrete action for
        the current state.

        Notes:
            actor network must output a probability tensor, of shape
            (batch_size, action_dims), and has a sum of 1 for each row
            in dimension 1.

        Args:
            state: Current state.
            use_target: Whether to use the target network.

        Returns:
            Noisy action of shape ``[batch_size, 1]``.
            Action probability tensor of shape ``[batch_size, action_num]``.
            Any other things returned by your Q network. if they exist.
        """
        if use_target:
            action, *others = safe_call(self.actor_target, state)
        else:
            action, *others = safe_call(self.actor, state)

        assert_output_is_probs(action)
        dist = Categorical(action)
        batch_size = action.shape[0]
        return (dist.sample([batch_size, 1]).view(batch_size, 1),
                *others)

    def _act(self,
             state: Dict[str, Any],
             use_target: bool = False,
             **__):
        """
        Use actor network to produce an action for the current state.

        Args:
            state: Current state.
            use_target: Whether use the target network.

        Returns:
            Action of shape ``[batch_size, action_dim]``.
        """
        if use_target:
            return safe_call(self.actor_target, state)[0]
        else:
            return safe_call(self.actor, state)[0]

    def _criticize(self,
                   state: Dict[str, Any],
                   action: Dict[str, Any],
                   use_target: bool = False,
                   **__):
        """
        Use critic network to evaluate current value.

        Args:
            state: Current state.
            action: Current action.
            use_target: Whether to use the target network.

        Returns:
            Q Value of shape ``[batch_size, 1]``.
        """
        if use_target:
            return safe_call(self.critic_target, state, action)[0]
        else:
            return safe_call(self.critic, state, action)[0]

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer.
        """
        self.replay_buffer.append(transition, required_attrs=(
            "state", "action", "reward", "next_state", "terminal"
        ))

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.
        """
        for trans in episode:
            self.replay_buffer.append(trans, required_attrs=(
                "state", "action", "reward", "next_state", "terminal"
            ))

    def update(self,
               update_value=True,
               update_policy=True,
               update_target=True,
               concatenate_samples=True,
               **__):
        """
        Update network weights by sampling from replay buffer.

        Args:
            update_value: Whether to update the Q network.
            update_policy: Whether to update the actor network.
            update_target: Whether to update targets.
            concatenate_samples: Whether to concatenate the samples.

        Returns:
            mean value of estimated policy value, value loss
        """
        self.actor.train()
        self.critic.train()
        batch_size, (state, action, reward, next_state, terminal, others) = \
            self.replay_buffer.sample_batch(self.batch_size,
                                            concatenate_samples,
                                            sample_method="random_unique",
                                            sample_attrs=[
                                                "state", "action",
                                                "reward", "next_state",
                                                "terminal", "*"
                                            ])

        # Update critic network first.
        # Generate value reference :math: `y_i` using target actor and
        # target critic.
        with t.no_grad():
            next_action = self.action_transform_function(
                self._act(next_state, True), next_state, others
            )
            next_value = self._criticize(next_state, next_action, True)
            next_value = next_value.view(batch_size, -1)
            y_i = self.reward_function(
                reward, self.discount, next_value, terminal, others
            )

        cur_value = self._criticize(state, action)
        value_loss = self.criterion(cur_value, y_i.to(cur_value.device))

        if self.visualize:
            self.visualize_model(value_loss, "critic", self.visualize_dir)

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.grad_max
            )
            self.critic_optim.step()

        # Update actor network
        cur_action = self.action_transform_function(
            self._act(state), state, others
        )
        act_value, *_ = self._criticize(state, cur_action)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        if self.visualize:
            self.visualize_model(act_policy_loss, "actor", self.visualize_dir)

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_max
            )
            self.actor_optim.step()

        # Update target networks
        if update_target:
            soft_update(self.actor_target, self.actor, self.update_rate)
            soft_update(self.critic_target, self.critic, self.update_rate)

        self.actor.eval()
        self.critic.eval()
        # use .item() to prevent memory leakage
        return -act_policy_loss.item(), value_loss.item()

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()

    def load(self, model_dir: str, network_map: Dict[str, str] = None,
             version: int = -1):
        # DOC INHERITED
        super(DDPG, self).load(model_dir, network_map, version)
        with t.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)

    @staticmethod
    def action_transform_function(raw_output_action: Any, *_):
        """
        The action transform function is used to transform the output
        of actor to the input of critic.
        Action transform function must accept:

          1. Raw action from the actor model.
          2. Concatenated :attr:`.Transition.next_state`.
          3. Any other concatenated lists of custom keys from \
              :class:`.Transition`.

        and returns:
          1. A dictionary with the same form as :attr:`.Transition.action`

        Args:
          raw_output_action: Raw action from the actor model.
        """
        return {"action": raw_output_action}

    @staticmethod
    def reward_function(reward, discount, next_value, terminal, _):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * ~terminal * next_value
