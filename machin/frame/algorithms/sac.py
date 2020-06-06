from typing import Union, Dict, List, Tuple, Callable, Any
import torch as t
import torch.nn as nn
import numpy as np

from machin.frame.buffers.buffer import Transition, Buffer
from machin.model.nets.base import NeuralNetworkModule
from .base import TorchFramework
from .utils import hard_update, soft_update, safe_call


class SAC(TorchFramework):
    """
    SAC framework.
    """

    _is_top = ["actor", "critic", "critic2", "critic_target", "critic2_target"]
    _is_restorable = ["actor", "critic_target", "critic2_target"]

    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 critic2: Union[NeuralNetworkModule, nn.Module],
                 critic2_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 criterion: Callable,
                 *_,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple] = (),
                 lr_scheduler_kwargs: Tuple[Dict, Dict] = (),
                 target_entropy: float = None,
                 initial_entropy_alpha: float = 1.0,
                 gradient_max: float = np.inf,
                 batch_size: int = 100,
                 update_rate: float = 0.005,
                 learning_rate: float = 0.001,
                 discount: float = 0.99,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 reward_func: Callable = None,
                 action_trans_func: Callable = None,
                 visualize: bool = False,
                 **__):
        """
        See Also:
            :class:`.A2C`
            :class:`.DDPG`

        Important:
            when given a state, and an optional, action actor must
            at least return two values, similar to the actor structure
            described in :class:`.A2C`. You may use the same actor model
            here.

            However, Compared to other actor-critic methods, SAC embeds the
            entropy term into its reward function directly, rather than adding
            the entropy term to actor's loss function. Therefore, we do not need
            the actor network to output a entropy as its third returned result.

            The SAC algorithm uses Q network as critics, so please reference
            :class:`.DDPG` for the requirements and the definition of
            ``action_trans_func``.

        Args:
            actor: Actor network module.
            critic: Critic network module.
            critic_target: Target critic network module.
            critic2: The second critic network module.
            critic2_target: The second target critic network module.
            optimizer: Optimizer used to optimize ``actor``, ``critic`` and
                ``critic2``.
            criterion: Criterion used to evaluate the value loss.
            *_:
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            target_entropy: Target entropy weight :math:`\\alpha` used in
                the SAC soft value function:
                :math:`V_{soft}(s_t) = \\mathbb{E}_{q_t\\sim\\pi}[\
                                        Q_{soft}(s_t,a_t) - \
                                        \\alpha log\\pi(a_t|s_t)]`
            initial_entropy_alpha: Initial entropy weight :math:`\\alpha`
            gradient_max: Maximum gradient.
            batch_size: Batch size used during training.
            update_rate: :math:`\\tau` used to update target networks.
                Target parameters are updated as:

                :math:`\\theta_t = \\theta * \\tau + \\theta_t * (1 - \\tau)`
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            discount: :math:`\\gamma` used in the bellman function.
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            reward_func: Reward function used in training.
            action_trans_func: Action transform function, used to transform
                the raw output of your actor, by default it is:
                ``lambda act: {"action": act}``
            visualize: Whether visualize the network flow in the first pass.
            **__:
        """
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.visualize = visualize
        self.entropy_alpha = t.tensor(initial_entropy_alpha,
                                      requires_grad=True).view(1)
        self.grad_max = gradient_max
        self.target_entropy = target_entropy

        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.critic2 = critic2
        self.critic2_target = critic2_target
        self.actor_optim = optimizer(self.actor.parameters(),
                                     lr=learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(),
                                      lr=learning_rate)
        self.critic2_optim = optimizer(self.critic2.parameters(),
                                       lr=learning_rate)
        self.alpha_optim = optimizer([self.entropy_alpha],
                                     lr=learning_rate)
        self.replay_buffer = (Buffer(replay_size, replay_device)
                              if replay_buffer is None
                              else replay_buffer)

        # Make sure target and online networks have the same weight
        with t.no_grad():
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic2, self.critic2_target)

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
            self.critic2_lr_sch = lr_scheduler(
                self.critic2_optim,
                *lr_scheduler_args[1],
                **lr_scheduler_kwargs[1]
            )

        self.criterion = criterion

        self.reward_func = (SAC.bellman_function
                            if reward_func is None
                            else reward_func)
        self.action_trans_func = (SAC.action_transform_function
                                  if action_trans_func is None
                                  else action_trans_func)

        super(SAC, self).__init__()

    def act(self, state: Dict[str, Any], **__):
        """
        Use actor network to produce an action for the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(self.actor, state)

    def criticize(self,
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
            Value evaluated by critic.
        """
        if use_target:
            return safe_call(self.critic_target, state, action)
        else:
            return safe_call(self.critic, state, action)

    def criticize2(self,
                   state: Dict[str, Any],
                   action: Dict[str, Any],
                   use_target: bool = False,
                   **__):
        """
        Use the second critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        """
        if use_target:
            return safe_call(self.critic2_target, state, action)
        else:
            return safe_call(self.critic2, state, action)

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer.
        """
        self.replay_buffer.append(transition, required_attrs=(
            "state", "action", "next_state", "reward", "terminal"
        ))

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.
        """
        for trans in episode:
            self.replay_buffer.append(trans, required_attrs=(
                "state", "action", "next_state", "reward", "terminal"
            ))

    def update(self,
               update_value=True,
               update_policy=True,
               update_target=True,
               update_entropy_alpha=True,
               concatenate_samples=True,
               **__):
        """
        Update network weights by sampling from replay buffer.

        Args:
            update_value: Whether to update the Q network.
            update_policy: Whether to update the actor network.
            update_target: Whether to update targets.
            update_entropy_alpha: Whether to update :math:`alpha` of entropy.
            concatenate_samples: Whether to concatenate the samples.

        Returns:
            mean value of estimated policy value, value loss
        """
        batch_size, (state, action, reward, next_state, terminal, *others) = \
            self.replay_buffer.sample_batch(self.batch_size,
                                            concatenate_samples,
                                            sample_attrs=[
                                                "state", "action",
                                                "reward", "next_state",
                                                "terminal", "*"
                                            ])

        # Update critic network first
        with t.no_grad():
            next_action, next_action_log_prob, _ = self.actor(next_state)
            next_action = self.action_trans_func(next_action, next_state,
                                                 *others)
            next_value = self.criticize(next_state, next_action, True)
            next_value2 = self.criticize2(next_state, next_action, True)
            next_value = t.min(next_value, next_value2)
            next_value = (next_value.view(batch_size, -1) -
                          self.entropy_alpha
                          * next_action_log_prob.view(batch_size, -1))
            y_i = self.reward_func(reward, self.discount, next_value,
                                   terminal, *others)

        cur_value = self.criticize(state, action)
        cur_value2 = self.criticize2(state, action)
        value_loss = self.criterion(cur_value, y_i.to(cur_value.device))
        value_loss2 = self.criterion(cur_value2, y_i.to(cur_value.device))

        if self.visualize:
            self.visualize_model(value_loss, "critic")

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
        act_policy_loss = ((self.entropy_alpha * cur_action_log_prob) -
                           act_value).mean()

        if self.visualize:
            self.visualize_model(act_policy_loss, "actor")

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            self.actor_optim.step()

        # Update target networks
        if update_target:
            soft_update(self.critic_target, self.critic, self.update_rate)

        if update_entropy_alpha:
            alpha_loss = -(t.log(self.entropy_alpha) *
                           (cur_action_log_prob + self.target_entropy).detach()
                           ).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

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

    def load(self, model_dir, network_map=None, version=-1):
        # DOC INHERITED
        super(SAC, self).load(model_dir, network_map, version)
        with t.no_grad():
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic, self.critic2_target)

    @staticmethod
    def action_transform_function(raw_output_action, *_):
        return {"action": raw_output_action}

    @staticmethod
    def bellman_function(reward, discount, next_value, terminal, *_):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * (1 - terminal) * next_value
