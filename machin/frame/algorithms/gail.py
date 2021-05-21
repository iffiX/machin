from typing import Union, Dict, List, Tuple, Callable, Any
from copy import deepcopy
import os
import torch as t
import torch.nn as nn
import numpy as np

from machin.model.nets.base import NeuralNetworkModule
from machin.frame.buffers.buffer import TransitionBase, Transition, Buffer
from .ppo import PPO
from .trpo import TRPO
from .base import TorchFramework, Config
from .utils import (
    safe_call,
    assert_and_get_valid_models,
    assert_and_get_valid_optimizer,
    assert_and_get_valid_lr_scheduler,
)


class ExpertTransition(TransitionBase):
    """
    The ExpertTransition class for expert steps.

    Have two main attributes: ``state`` and ``action``.
    """

    # for auto suggestion in IDEs

    state = None  # type: Dict[str, t.Tensor]
    action = None  # type: Dict[str, t.Tensor]

    def __init__(
        self, state: Dict[str, t.Tensor], action: Dict[str, t.Tensor],
    ):
        """
        Args:
            state: Previous observed state.
            action: Action of expert.
        """
        super().__init__(
            major_attr=["state", "action"],
            sub_attr=[],
            custom_attr=[],
            major_data=[state, action],
            sub_data=[],
            custom_data=[],
        )

    def _check_validity(self):
        # fix batch size to 1
        super()._check_validity()
        if self._batch_size != 1:
            raise ValueError(
                "Batch size of the expert transition "
                f"implementation must be 1, is {self._batch_size}"
            )


class GAIL(TorchFramework):
    """
    GAIL framework.
    """

    _is_top = ["actor", "critic", "discriminator"]
    _is_restorable = ["actor", "critic", "discriminator"]

    def __init__(
        self,
        discriminator: Union[NeuralNetworkModule, nn.Module],
        constrained_policy_optimization: Union[PPO, TRPO],
        optimizer: Callable,
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple] = None,
        lr_scheduler_kwargs: Tuple[Dict] = None,
        batch_size: int = 100,
        discriminator_update_times: int = 1,
        discriminator_learning_rate: float = 0.001,
        gradient_max: float = np.inf,
        expert_replay_size: int = 500000,
        expert_replay_device: Union[str, t.device] = "cpu",
        expert_replay_buffer: Buffer = None,
        visualize: bool = False,
        visualize_dir: str = "",
        **__,
    ):
        """
        Note:
            The forward method of your discriminator network must take two arguments::
                
                def forward(self,
                            state: Dict[str, t.Tensor],
                            action: Dict[str, t.Tensor])

            And return a tag vector (float type) of size ``[batch_size, 1]``, usually
            you can do this by using a sigmoid output layer.

            If you set ``concatenate_samples`` to ``False`` during the ``update()``
            call, then you should expect ``Dict[str, List[t.Tensor]]``.

        Note:
            You can access the following attributes:

                1. ``actor``
                2. ``critic``
                3. ``actor_optim``
                4. ``critic_optim``
                5. ``actor_lr_sch`` (may not exist if you are not using lr scheduler)
                6. ``critic_lr_sch`` (may not exist if you are not using lr scheduler)
                7. ``replay_buffer``

            of the input PPO or TRPO framework directly from an GAIL instance,
            since they are forwarded to here.

            For other attributes, you need to manually access them from the
            ``constrained_policy_optimization`` attribute.

        Args:
            discriminator: Discriminator network module.
            constrained_policy_optimization: A constrained policy optimization
                framework, currently can be a :class:`.PPO` or :class:`TRPO` framework.
            optimizer: Optimizer used to optimize ``discriminator``.
            discriminator_learning_rate: Learning rate of the discriminator optimizer,
                not compatible with ``lr_scheduler``.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used during discriminator training.
            gradient_max: Maximum gradient.
            expert_replay_size: Expert trajectory buffer size. Not compatible with
                ``expert_replay_buffer``.
            expert_replay_device: Device where the expert replay buffer locates on, Not
                compatible with ``expert_replay_buffer``.
            expert_replay_buffer: Custom expert replay buffer.
            visualize: Whether visualize the network flow in the first pass.
            visualize_dir: Visualized graph save directory.
        """
        self.batch_size = batch_size
        self.gradient_max = gradient_max
        self.visualize = visualize
        self.visualize_dir = visualize_dir

        self.constrained_policy_optimization = constrained_policy_optimization
        self.actor = constrained_policy_optimization.actor
        self.critic = constrained_policy_optimization.critic
        self.actor_optim = constrained_policy_optimization.actor_optim
        self.critic_optim = constrained_policy_optimization.critic_optim
        if hasattr(constrained_policy_optimization, "actor_lr_sch"):
            self.actor_lr_sch = constrained_policy_optimization.actor_lr_sch
        if hasattr(constrained_policy_optimization, "critic_lr_sch"):
            self.critic_lr_sch = constrained_policy_optimization.critic_lr_sch
        self.replay_buffer = constrained_policy_optimization.replay_buffer

        self.discriminator = discriminator
        # By default it is BCELoss, you can modify this attribute.
        self.discriminator_criterion = nn.BCELoss()
        self.discriminator_update_times = discriminator_update_times
        self.discriminator_optim = optimizer(
            self.actor.parameters(), lr=discriminator_learning_rate
        )
        self.expert_replay_buffer = (
            Buffer(expert_replay_size, expert_replay_device)
            if expert_replay_buffer is None
            else expert_replay_buffer
        )

        if lr_scheduler is not None:
            if lr_scheduler_args is None:
                lr_scheduler_args = ((),)
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = ({},)
            self.discriminator_lr_sch = lr_scheduler(
                self.discriminator_optim,
                *lr_scheduler_args[0],
                **lr_scheduler_kwargs[0],
            )

        self.bce_criterion = nn.BCELoss()
        super().__init__()

    @property
    def optimizers(self):
        return [self.actor_optim, self.critic_optim, self.discriminator_optim]

    @optimizers.setter
    def optimizers(self, optimizers):
        self.actor_optim, self.critic_optim, self.discriminator_optim = optimizers

    @property
    def lr_schedulers(self):
        lr_schs = []
        if hasattr(self, "actor_lr_sch") and hasattr(self, "critic_lr_sch"):
            lr_schs += [self.actor_lr_sch, self.critic_lr_sch]
        if hasattr(self, "discriminator_lr_sch"):
            lr_schs += [self.discriminator_lr_sch]
        return lr_schs

    def act(self, state: Dict[str, Any], *_, **__):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Anything produced by actor.
        """
        # No need to safe_return because the number of
        # returned values is always more than one
        return safe_call(self.actor, state)

    def _discriminate(self, state: Dict[str, Any], action: Dict[str, Any], *_, **__):
        """
        Use discriminator network to assign a real (0) / fake (1) tag to state-action
        pairs.

        Returns:
            Tags of shape ``[batch_size, 1]``
        """
        return safe_call(self.discriminator, state, action)[0]

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.

        "value" and "gae" are automatically calculated.
        """
        # replace reward with expert reward
        # don't do pre-concatenation since state size may be uneven.
        for trans in episode:
            trans["reward"] = -np.log(self._discriminate(**trans).item())
        self.constrained_policy_optimization.store_episode(episode)

    def store_expert_episode(self, episode: List[Union[ExpertTransition, Dict]]):
        """
        Add a full episode of transition samples from the expert trajectory
        to the replay buffer.

        Only states and actions are required.
        """
        for trans in episode:
            if isinstance(trans, dict):
                trans = ExpertTransition(**trans)
            self.expert_replay_buffer.append(trans, required_attrs=("state", "action"))

    def update(
        self,
        update_value=True,
        update_policy=True,
        update_discriminator=True,
        concatenate_samples=True,
        **__,
    ):
        """
        Update network weights by sampling from buffer. Buffer
        will be cleared after update is finished.

        Args:
            update_value: Whether update the Q network.
            update_policy: Whether update the actor network.
            update_discriminator: Whether update the discriminator network.
            concatenate_samples: Whether concatenate the samples.

        Returns:
            mean value of estimated policy value, value loss
        """
        self.actor.train()
        self.critic.train()
        self.discriminator.train()
        sum_discrim_loss = 0

        for _ in range(self.discriminator_update_times):
            # sample a batch from expert buffer and a batch from training buffer.
            e_batch_size, (e_state, e_action) = self.expert_replay_buffer.sample_batch(
                self.batch_size,
                sample_method="random_unique",
                concatenate=concatenate_samples,
                sample_attrs=["state", "action"],
            )
            exp_out = self._discriminate(e_state, e_action)

            batch_size, (state, action) = self.replay_buffer.sample_batch(
                self.batch_size,
                sample_method="random_unique",
                concatenate=concatenate_samples,
                sample_attrs=["state", "action"],
            )
            gen_out = self._discriminate(state, action)
            discrim_loss = self.discriminator_criterion(
                gen_out, t.ones_like(gen_out)
            ) + self.discriminator_criterion(exp_out, t.zeros_like(exp_out))

            # Update discriminator network
            if update_discriminator:
                self.discriminator.zero_grad()
                self._backward(discrim_loss)
                nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.gradient_max
                )
                self.discriminator_optim.step()

            if self.visualize:
                self.visualize_model(discrim_loss, "discriminator", self.visualize_dir)

            sum_discrim_loss += discrim_loss.item()

        # perform mini-batch PPO or TRPO update
        act_loss, value_loss = self.constrained_policy_optimization.update(
            update_value=update_value,
            update_policy=update_policy,
            concatenate_samples=concatenate_samples,
        )
        self.actor.eval()
        self.critic.eval()
        self.discriminator.eval()

        return act_loss, value_loss, sum_discrim_loss / self.discriminator_update_times

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()
        if hasattr(self, "discriminator_lr_sch"):
            self.discriminator_lr_sch.step()

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        default_values = {
            "models": ["Discriminator"],
            "model_args": ((),),
            "model_kwargs": ({},),
            "constrained_policy_optimization": "PPO",
            "optimizer": "Adam",
            "PPO_config": PPO.generate_config({}),
            "TRPO_config": TRPO.generate_config({}),
            "lr_scheduler": None,
            "lr_scheduler_args": None,
            "lr_scheduler_kwargs": None,
            "batch_size": 100,
            "discriminator_update_times": 1,
            "discriminator_learning_rate": 0.001,
            "gradient_max": np.inf,
            "expert_trajectory_path": "trajectory.data",
            "expert_replay_size": 500000,
            "expert_replay_device": "cpu",
            "expert_replay_buffer": None,
            "visualize": False,
            "visualize_dir": "",
        }
        config = deepcopy(config)
        config["frame"] = "GAIL"
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
        # Initialize PPO or TRPO first
        if f_config["constrained_policy_optimization"] == "PPO":
            cpo = PPO.init_from_config(f_config["PPO_config"], model_device)
        elif f_config["constrained_policy_optimization"] == "TRPO":
            cpo = TRPO.init_from_config(f_config["TRPO_config"], model_device)
        else:
            raise ValueError("constrained_policy_optimization must be PPO or TRPO.")

        discrim_model = assert_and_get_valid_models(f_config["models"])[0]
        discrim_model_args = f_config["model_args"][0]
        discrim_model_kwargs = f_config["model_kwargs"][0]
        discrim = discrim_model(*discrim_model_args, **discrim_model_kwargs)
        optimizer = assert_and_get_valid_optimizer(f_config["optimizer"])
        lr_scheduler = f_config["lr_scheduler"] and assert_and_get_valid_lr_scheduler(
            f_config["lr_scheduler"]
        )

        f_config["constrained_policy_optimization"] = cpo
        f_config["optimizer"] = optimizer
        f_config["lr_scheduler"] = lr_scheduler
        frame = cls(discrim, **f_config)

        if os.path.isfile(f_config["expert_trajectory_path"]):
            trajectory_list = t.load(f_config["expert_trajectory_path"])
            for trajectory in trajectory_list:
                frame.store_expert_episode(trajectory)
        return frame
