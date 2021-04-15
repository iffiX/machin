from typing import Union, Dict, List, Tuple, Callable, Any
from copy import deepcopy
import torch as t
import torch.nn as nn
import numpy as np

from machin.frame.buffers.buffer import Transition, Buffer
from machin.model.nets.base import NeuralNetworkModule
from .base import TorchFramework, Config
from .utils import (
    hard_update,
    soft_update,
    safe_call,
    safe_return,
    assert_and_get_valid_models,
    assert_and_get_valid_optimizer,
    assert_and_get_valid_criterion,
    assert_and_get_valid_lr_scheduler,
)


class SAC(TorchFramework):
    """
    SAC framework.
    """

    _is_top = ["actor", "critic", "critic2", "critic_target", "critic2_target"]
    _is_restorable = ["actor", "critic_target", "critic2_target"]

    def __init__(
        self,
        actor: Union[NeuralNetworkModule, nn.Module],
        critic: Union[NeuralNetworkModule, nn.Module],
        critic_target: Union[NeuralNetworkModule, nn.Module],
        critic2: Union[NeuralNetworkModule, nn.Module],
        critic2_target: Union[NeuralNetworkModule, nn.Module],
        optimizer: Callable,
        criterion: Callable,
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple, Tuple, Tuple] = None,
        lr_scheduler_kwargs: Tuple[Dict, Dict, Dict] = None,
        target_entropy: float = None,
        initial_entropy_alpha: float = 1.0,
        batch_size: int = 100,
        update_rate: float = 0.005,
        update_steps: Union[int, None] = None,
        actor_learning_rate: float = 0.0005,
        critic_learning_rate: float = 0.001,
        alpha_learning_rate: float = 0.001,
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
        See Also:
            :class:`.A2C`
            :class:`.DDPG`

        Important:
            When given a state, and an optional action, actor must
            at least return two values, similar to the actor structure
            described in :class:`.A2C`. However, when actor is asked to
            select an action based on the current state, you must make
            sure that the sampling process is **differentiable**. E.g.
            use the ``rsample`` method of torch distributions instead
            of the ``sample`` method.

            Compared to other actor-critic methods, SAC embeds the
            entropy term into its reward function directly, rather than adding
            the entropy term to actor's loss function. Therefore, we do not use
            the entropy output of your actor network.

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
        self.visualize = visualize
        self.visualize_dir = visualize_dir
        self.entropy_alpha = t.tensor([initial_entropy_alpha], requires_grad=True)
        self.grad_max = gradient_max
        self.target_entropy = target_entropy
        self._update_counter = 0

        if update_rate is not None and update_steps is not None:
            raise ValueError(
                "You can only specify one target network update"
                " scheme, either by update_rate or update_steps,"
                " but not both."
            )

        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.critic2 = critic2
        self.critic2_target = critic2_target
        self.actor_optim = optimizer(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), lr=critic_learning_rate)
        self.critic2_optim = optimizer(
            self.critic2.parameters(), lr=critic_learning_rate
        )
        self.alpha_optim = optimizer([self.entropy_alpha], lr=alpha_learning_rate)
        self.replay_buffer = (
            Buffer(replay_size, replay_device)
            if replay_buffer is None
            else replay_buffer
        )

        # Make sure target and online networks have the same weight
        with t.no_grad():
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic2, self.critic2_target)

        if lr_scheduler is not None:
            if lr_scheduler_args is None:
                lr_scheduler_args = ((), (), ())
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = ({}, {}, {})
            self.actor_lr_sch = lr_scheduler(
                self.actor_optim, *lr_scheduler_args[0], **lr_scheduler_kwargs[0],
            )
            self.critic_lr_sch = lr_scheduler(
                self.critic_optim, *lr_scheduler_args[1], **lr_scheduler_kwargs[1]
            )
            self.critic2_lr_sch = lr_scheduler(
                self.critic2_optim, *lr_scheduler_args[1], **lr_scheduler_kwargs[1]
            )
            self.alpha_lr_sch = lr_scheduler(
                self.alpha_optim, *lr_scheduler_args[2], **lr_scheduler_kwargs[2]
            )

        self.criterion = criterion
        super().__init__()

    @property
    def optimizers(self):
        return [
            self.actor_optim,
            self.critic_optim,
            self.critic2_optim,
            self.alpha_optim,
        ]

    @optimizers.setter
    def optimizers(self, optimizers):
        (
            self.actor_optim,
            self.critic_optim,
            self.critic2_optim,
            self.alpha_optim,
        ) = optimizers

    @property
    def lr_schedulers(self):
        if (
            hasattr(self, "actor_lr_sch")
            and hasattr(self, "critic_lr_sch")
            and hasattr(self, "critic2_lr_sch")
            and hasattr(self, "alpha_lr_sch")
        ):
            return [
                self.actor_lr_sch,
                self.critic_lr_sch,
                self.critic2_lr_sch,
                self.alpha_lr_sch,
            ]
        return []

    def act(self, state: Dict[str, Any], **__):
        """
        Use actor network to produce an action for the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_return(safe_call(self.actor, state))

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

    def _criticize2(
        self, state: Dict[str, Any], action: Dict[str, Any], use_target=False, **__
    ):
        """
        Use the second critic network to evaluate current value.

        Args:
            state: Current state.
            action: Current action.
            use_target: Whether to use the target network.

        Returns:
            Q Value of shape ``[batch_size, 1]``.
        """
        if use_target:
            return safe_call(self.critic2_target, state, action)[0]
        else:
            return safe_call(self.critic2, state, action)[0]

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer.
        """
        self.replay_buffer.append(
            transition,
            required_attrs=("state", "action", "next_state", "reward", "terminal"),
        )

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.
        """
        for trans in episode:
            self.replay_buffer.append(
                trans,
                required_attrs=("state", "action", "next_state", "reward", "terminal"),
            )

    def update(
        self,
        update_value=True,
        update_policy=True,
        update_target=True,
        update_entropy_alpha=True,
        concatenate_samples=True,
        **__
    ):
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
        self.actor.train()
        self.critic.train()
        self.critic2.train()
        (
            batch_size,
            (state, action, reward, next_state, terminal, others,),
        ) = self.replay_buffer.sample_batch(
            self.batch_size,
            concatenate_samples,
            sample_method="random_unique",
            sample_attrs=["state", "action", "reward", "next_state", "terminal", "*"],
        )

        # Update critic network first
        with t.no_grad():
            next_action, next_action_log_prob, *_ = self.act(next_state)
            next_action = self.action_transform_function(
                next_action, next_state, others
            )
            next_value = self._criticize(next_state, next_action, True)
            next_value2 = self._criticize2(next_state, next_action, True)
            next_value = t.min(next_value, next_value2)
            next_value = next_value.view(
                batch_size, -1
            ) - self.entropy_alpha.item() * next_action_log_prob.view(batch_size, -1)
            y_i = self.reward_function(
                reward, self.discount, next_value, terminal, others
            )

        cur_value = self._criticize(state, action)
        cur_value2 = self._criticize2(state, action)
        value_loss = self.criterion(cur_value, y_i.type_as(cur_value))
        value_loss2 = self.criterion(cur_value2, y_i.type_as(cur_value))

        if self.visualize:
            self.visualize_model(value_loss, "critic", self.visualize_dir)

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_max)
            self.critic_optim.step()

            self.critic2.zero_grad()
            value_loss2.backward()
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_max)
            self.critic2_optim.step()

        # Update actor network
        cur_action, cur_action_log_prob, *_ = self.act(state)
        cur_action = self.action_transform_function(cur_action, state, others)
        act_value = self._criticize(state, cur_action)
        act_value2 = self._criticize2(state, cur_action)
        act_value = t.min(act_value, act_value2)

        act_policy_loss = (
            self.entropy_alpha.item() * cur_action_log_prob - act_value
        ).mean()

        if self.visualize:
            self.visualize_model(act_policy_loss, "actor", self.visualize_dir)

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_max)
            self.actor_optim.step()

        # Update target networks
        if update_target:
            if self.update_rate is not None:
                soft_update(self.critic_target, self.critic, self.update_rate)
                soft_update(self.critic2_target, self.critic2, self.update_rate)
            else:
                self._update_counter += 1
                if self._update_counter % self.update_steps == 0:
                    hard_update(self.critic_target, self.critic)
                    hard_update(self.critic2_target, self.critic2)

        if update_entropy_alpha and self.target_entropy is not None:
            alpha_loss = -(
                t.log(self.entropy_alpha)
                * (cur_action_log_prob + self.target_entropy).cpu().detach()
            ).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            # prevent nan
            with t.no_grad():
                self.entropy_alpha.clamp_(1e-6, 1e6)

        self.actor.eval()
        self.critic.eval()
        self.critic2.eval()
        # use .item() to prevent memory leakage
        return (-act_policy_loss.item(), (value_loss.item() + value_loss2.item()) / 2)

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
        super().load(model_dir, network_map, version)
        with t.no_grad():
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic, self.critic2_target)

    @staticmethod
    def action_transform_function(raw_output_action, *_):
        return {"action": raw_output_action}

    @staticmethod
    def reward_function(reward, discount, next_value, terminal, _):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * ~terminal * next_value

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        default_values = {
            "models": ["Actor", "Critic", "Critic", "Critic", "Critic"],
            "model_args": ((), (), (), (), ()),
            "model_kwargs": ({}, {}, {}, {}, {}),
            "optimizer": "Adam",
            "criterion": "MSELoss",
            "criterion_args": (),
            "criterion_kwargs": {},
            "lr_scheduler": None,
            "lr_scheduler_args": None,
            "lr_scheduler_kwargs": None,
            "target_entropy": None,
            "initial_entropy_alpha": 1.0,
            "batch_size": 100,
            "update_rate": 0.001,
            "update_steps": None,
            "actor_learning_rate": 0.0005,
            "critic_learning_rate": 0.001,
            "alpha_learning_rate": 0.001,
            "discount": 0.99,
            "gradient_max": np.inf,
            "replay_size": 500000,
            "replay_device": "cpu",
            "replay_buffer": None,
            "visualize": False,
            "visualize_dir": "",
        }
        config = deepcopy(config)
        config["frame"] = "SAC"
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
