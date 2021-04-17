from typing import Union, Dict, List, Tuple, Callable, Any
from copy import deepcopy
from torch.distributions import Categorical
import torch as t
import torch.nn as nn
import numpy as np

from machin.frame.buffers.buffer import Transition, Buffer
from machin.frame.noise.action_space_noise import (
    add_normal_noise_to_action,
    add_clipped_normal_noise_to_action,
    add_uniform_noise_to_action,
    add_ou_noise_to_action,
)
from machin.model.nets.base import NeuralNetworkModule
from .base import TorchFramework, Config
from .utils import (
    hard_update,
    soft_update,
    safe_call,
    safe_return,
    assert_output_is_probs,
    assert_and_get_valid_models,
    assert_and_get_valid_optimizer,
    assert_and_get_valid_criterion,
    assert_and_get_valid_lr_scheduler,
)


class DDPG(TorchFramework):
    """
    DDPG framework.
    """

    _is_top = ["actor", "critic", "actor_target", "critic_target"]
    _is_restorable = ["actor_target", "critic_target"]

    def __init__(
        self,
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
        update_steps: Union[int, None] = None,
        actor_learning_rate: float = 0.0005,
        critic_learning_rate: float = 0.001,
        discount: float = 0.99,
        gradient_max: float = np.inf,
        replay_size: int = 500000,
        replay_device: Union[str, t.device] = "cpu",
        replay_buffer: Buffer = None,
        visualize: bool = False,
        visualize_dir: str = "",
        **__
    ):
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

        Note:
            DDPG supports two ways of updating the target network, the first
            way is polyak update (soft update), which updates the target network
            in every training step by mixing its weights with the online network
            using ``update_rate``.

            The other way is hard update, which copies weights of the online
            network after every ``update_steps`` training step.

            You can either specify ``update_rate`` or ``update_steps`` to select
            one update scheme, if both are specified, an error will be raised.

            These two different update schemes may result in different training
            stability.

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
            update_steps: Training step number used to update target networks.
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
        self.update_steps = update_steps
        self.discount = discount
        self.gradient_max = gradient_max
        self.visualize = visualize
        self.visualize_dir = visualize_dir
        self._update_counter = 0

        if update_rate is not None and update_steps is not None:
            raise ValueError(
                "You can only specify one target network update"
                " scheme, either by update_rate or update_steps,"
                " but not both."
            )

        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optim = optimizer(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), lr=critic_learning_rate)
        self.replay_buffer = (
            Buffer(replay_size, replay_device)
            if replay_buffer is None
            else replay_buffer
        )

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
                self.actor_optim, *lr_scheduler_args[0], **lr_scheduler_kwargs[0]
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

    def act(self, state: Dict[str, Any], use_target: bool = False, **__):
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

    def act_with_noise(
        self,
        state: Dict[str, Any],
        noise_param: Any = (0.0, 1.0),
        ratio: float = 1.0,
        mode: str = "uniform",
        use_target: bool = False,
        **__
    ):
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
            noisy_action = add_uniform_noise_to_action(action, noise_param, ratio)
        elif mode == "normal":
            noisy_action = add_normal_noise_to_action(action, noise_param, ratio)
        elif mode == "clipped_normal":
            noisy_action = add_clipped_normal_noise_to_action(
                action, noise_param, ratio
            )
        elif mode == "ou":
            noisy_action = add_ou_noise_to_action(action, noise_param, ratio)
        else:
            raise ValueError("Unknown noise type: " + str(mode))

        if len(others) == 0:
            return noisy_action
        else:
            return (noisy_action, *others)

    def act_discrete(self, state: Dict[str, Any], use_target: bool = False, **__):
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

    def act_discrete_with_noise(
        self,
        state: Dict[str, Any],
        use_target: bool = False,
        choose_max_prob: float = 0.95,
        **__
    ):
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
            choose_max_prob: Probability to choose the largest component when actor
                is outputing extreme probability vector like ``[0, 1, 0, 0]``.

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
        batch_size = action.shape[0]
        action_dim = action.shape[1]
        if action_dim > 1 and choose_max_prob < 1.0:
            scale = np.log((action_dim - 1) / (1 - choose_max_prob) * choose_max_prob)
            action = t.softmax(action * scale, dim=1)
        dist = Categorical(action)

        result = dist.sample([batch_size, 1]).view(batch_size, 1)
        return (result, action, *others)

    def _act(self, state: Dict[str, Any], use_target: bool = False, **__):
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

    def _criticize(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        use_target: bool = False,
        **__
    ):
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
        self.replay_buffer.append(
            transition,
            required_attrs=("state", "action", "reward", "next_state", "terminal"),
        )

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.
        """
        for trans in episode:
            self.replay_buffer.append(
                trans,
                required_attrs=("state", "action", "reward", "next_state", "terminal"),
            )

    def update(
        self,
        update_value=True,
        update_policy=True,
        update_target=True,
        concatenate_samples=True,
        **__
    ):
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
        (
            batch_size,
            (state, action, reward, next_state, terminal, others,),
        ) = self.replay_buffer.sample_batch(
            self.batch_size,
            concatenate_samples,
            sample_method="random_unique",
            sample_attrs=["state", "action", "reward", "next_state", "terminal", "*"],
        )

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
        value_loss = self.criterion(cur_value, y_i.type_as(cur_value))

        if self.visualize:
            self.visualize_model(value_loss, "critic", self.visualize_dir)

        if update_value:
            self.critic.zero_grad()
            self._backward(value_loss)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_max)
            self.critic_optim.step()

        # Update actor network
        cur_action = self.action_transform_function(self._act(state), state, others)
        act_value, *_ = self._criticize(state, cur_action)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        if self.visualize:
            self.visualize_model(act_policy_loss, "actor", self.visualize_dir)

        if update_policy:
            self.actor.zero_grad()
            self._backward(act_policy_loss)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_max)
            self.actor_optim.step()

        # Update target networks
        if update_target:
            if self.update_rate is not None:
                soft_update(self.actor_target, self.actor, self.update_rate)
                soft_update(self.critic_target, self.critic, self.update_rate)
            else:
                self._update_counter += 1
                if self._update_counter % self.update_steps == 0:
                    hard_update(self.actor_target, self.actor)
                    hard_update(self.critic_target, self.critic)

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

    def load(
        self, model_dir: str, network_map: Dict[str, str] = None, version: int = -1
    ):
        # DOC INHERITED
        super().load(model_dir, network_map, version)
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

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        default_values = {
            "models": ["Actor", "Actor", "Critic", "Critic"],
            "model_args": ((), (), (), ()),
            "model_kwargs": ({}, {}, {}, {}),
            "optimizer": "Adam",
            "criterion": "MSELoss",
            "criterion_args": (),
            "criterion_kwargs": {},
            "lr_scheduler": None,
            "lr_scheduler_args": None,
            "lr_scheduler_kwargs": None,
            "batch_size": 100,
            "update_rate": 0.001,
            "update_steps": None,
            "actor_learning_rate": 0.0005,
            "critic_learning_rate": 0.001,
            "gradient_max": np.inf,
            "discount": 0.99,
            "replay_size": 500000,
            "replay_device": "cpu",
            "replay_buffer": None,
            "visualize": False,
            "visualize_dir": "",
        }
        config = deepcopy(config)
        config["frame"] = "DDPG"
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
        f_config = config["frame_config"]
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
