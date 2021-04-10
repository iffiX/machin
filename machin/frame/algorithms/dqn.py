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
    assert_and_get_valid_models,
    assert_and_get_valid_optimizer,
    assert_and_get_valid_criterion,
    assert_and_get_valid_lr_scheduler,
)


class DQN(TorchFramework):
    """
    DQN framework.
    """

    _is_top = ["qnet", "qnet_target"]
    _is_restorable = ["qnet_target"]

    def __init__(
        self,
        qnet: Union[NeuralNetworkModule, nn.Module],
        qnet_target: Union[NeuralNetworkModule, nn.Module],
        optimizer: Callable,
        criterion: Callable,
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple] = None,
        lr_scheduler_kwargs: Tuple[Dict] = None,
        batch_size: int = 100,
        epsilon_decay: float = 0.9999,
        update_rate: Union[float, None] = 0.005,
        update_steps: Union[int, None] = None,
        learning_rate: float = 0.001,
        discount: float = 0.99,
        gradient_max: float = np.inf,
        replay_size: int = 500000,
        replay_device: Union[str, t.device] = "cpu",
        replay_buffer: Buffer = None,
        mode: str = "double",
        visualize: bool = False,
        visualize_dir: str = "",
        **__,
    ):
        """
        Note:
            DQN is only available for discrete environments.

        Note:
            Dueling DQN is a network structure rather than a framework, so
            it could be applied to all three modes.

            If ``mode = "vanilla"``, implements the simplest online DQN,
            with replay buffer.

            If ``mode = "fixed_target"``, implements DQN with a target network,
            and replay buffer. Described in `this <https://web.stanford.\
edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf>`__ essay.

            If ``mode = "double"``, implements Double DQN described in
            `this <https://arxiv.org/pdf/1509.06461.pdf>`__ essay.

        Note:
            Vanilla DQN only needs one network, so internally, ``qnet``
            is assigned to ``qnet_target``.

        Note:
            In order to implement dueling DQN, you should create two dense
            output layers.

            In your q network::

                    self.fc_adv = nn.Linear(in_features=...,
                                            out_features=num_actions)
                    self.fc_val = nn.Linear(in_features=...,
                                            out_features=1)

            Then in your ``forward()`` method, you should implement output as::

                    adv = self.fc_adv(some_input)
                    val = self.fc_val(some_input).expand(self.batch_sze,
                                                         self.num_actions)
                    return val + adv - adv.mean(1, keepdim=True)

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
            DQN supports two ways of updating the target network, the first
            way is polyak update (soft update), which updates the target network
            in every training step by mixing its weights with the online network
            using ``update_rate``.

            The other way is hard update, which copies weights of the online
            network after every ``update_steps`` training step.

            You can either specify ``update_rate`` or ``update_steps`` to select
            one update scheme, if both are specified, an error will be raised.

            These two different update schemes may result in different training
            stability.

        Attributes:
            epsilon: Current epsilon value, determines randomness in
                ``act_discrete_with_noise``. You can set it to any value.

        Args:
            qnet: Q network module.
            qnet_target: Target Q network module.
            optimizer: Optimizer used to optimize ``qnet``.
            criterion: Criterion used to evaluate the value loss.
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used during training.
            epsilon_decay: Epsilon decay rate per acting with noise step.
                ``epsilon`` attribute is multiplied with this every time
                ``act_discrete_with_noise`` is called.
            update_rate: :math:`\\tau` used to update target networks.
                Target parameters are updated as:

                :math:`\\theta_t = \\theta * \\tau + \\theta_t * (1 - \\tau)`
            update_steps: Training step number used to update target networks.
            discount: :math:`\\gamma` used in the bellman function.
            gradient_max: Maximum gradient.
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            mode: one of ``"vanilla", "fixed_target", "double"``.
            visualize: Whether visualize the network flow in the first pass.
        """
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.update_rate = update_rate
        self.update_steps = update_steps
        self.discount = discount
        self.grad_max = gradient_max
        self.visualize = visualize
        self.visualize_dir = visualize_dir
        self.mode = mode
        self.epsilon = 1
        self._update_counter = 0

        if mode not in {"vanilla", "fixed_target", "double"}:
            raise ValueError(f"Unknown DQN mode: {mode}")

        if update_rate is not None and update_steps is not None:
            raise ValueError(
                "You can only specify one target network update"
                " scheme, either by update_rate or update_steps,"
                " but not both."
            )

        self.qnet = qnet
        if self.mode == "vanilla":
            self.qnet_target = qnet
        else:
            self.qnet_target = qnet_target
        self.qnet_optim = optimizer(self.qnet.parameters(), lr=learning_rate)
        self.replay_buffer = (
            Buffer(replay_size, replay_device)
            if replay_buffer is None
            else replay_buffer
        )

        # Make sure target and online networks have the same weight
        with t.no_grad():
            hard_update(self.qnet, self.qnet_target)

        if lr_scheduler is not None:
            if lr_scheduler_args is None:
                lr_scheduler_args = ((),)
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = ({},)
            self.qnet_lr_sch = lr_scheduler(
                self.qnet_optim, *lr_scheduler_args[0], **lr_scheduler_kwargs[0]
            )

        self.criterion = criterion

        super().__init__()

    @property
    def optimizers(self):
        return [self.qnet_optim]

    @optimizers.setter
    def optimizers(self, optimizers):
        self.qnet_optim = optimizers[0]

    @property
    def lr_schedulers(self):
        if hasattr(self, "qnet_lr_sch"):
            return [self.qnet_lr_sch]
        return []

    def act_discrete(self, state: Dict[str, Any], use_target: bool = False, **__):
        """
        Use Q network to produce a discrete action for
        the current state.

        Args:
            state: Current state.
            use_target: Whether to use the target network.

        Returns:
            Action of shape ``[batch_size, 1]``.
            Any other things returned by your Q network. if they exist.
        """
        if use_target:
            result, *others = safe_call(self.qnet_target, state)
        else:
            result, *others = safe_call(self.qnet, state)

        result = t.argmax(result, dim=1).view(-1, 1)
        if len(others) == 0:
            return result
        else:
            return (result, *others)

    def act_discrete_with_noise(
        self,
        state: Dict[str, Any],
        use_target: bool = False,
        decay_epsilon: bool = True,
        **__,
    ):
        """
        Randomly selects an action from the action space according
        to a uniform distribution, with regard to the epsilon decay
        policy.

        Args:
            state: Current state.
            use_target: Whether to use the target network.
            decay_epsilon: Whether to decay the ``epsilon`` attribute.

        Returns:
            Noisy action of shape ``[batch_size, 1]``.
            Any other things returned by your Q network. if they exist.
        """
        if use_target:
            result, *others = safe_call(self.qnet_target, state)
        else:
            result, *others = safe_call(self.qnet, state)

        action_dim = result.shape[1]
        result = t.argmax(result, dim=1).view(-1, 1)

        if t.rand([1]).item() < self.epsilon:
            result = t.randint(0, action_dim, [result.shape[0], 1])

        if decay_epsilon:
            self.epsilon *= self.epsilon_decay

        if len(others) == 0:
            return result
        else:
            return (result, *others)

    def _act_discrete(self, state: Dict[str, Any], use_target: bool = False, **__):
        """
        Use Q network to produce a discrete action for
        the current state.

        Args:
            state: Current state.
            use_target: Whether to use the target network.

        Returns:
            Action of shape ``[batch_size, 1]``
        """
        if use_target:
            result, *others = safe_call(self.qnet_target, state)
        else:
            result, *others = safe_call(self.qnet, state)
        return t.argmax(result, dim=1).view(-1, 1)

    def _criticize(self, state: Dict[str, Any], use_target: bool = False, **__):
        """
        Use Q network to evaluate current value.

        Args:
            state: Current state.
            use_target: Whether to use the target network.
        """
        if use_target:
            return safe_call(self.qnet_target, state)[0]
        else:
            return safe_call(self.qnet, state)[0]

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
        self, update_value=True, update_target=True, concatenate_samples=True, **__
    ):
        """
        Update network weights by sampling from replay buffer.

        Args:
            update_value: Whether update the Q network.
            update_target: Whether update targets.
            concatenate_samples: Whether concatenate the samples.

        Returns:
            mean value of estimated policy value, value loss
        """
        (
            batch_size,
            (state, action, reward, next_state, terminal, others,),
        ) = self.replay_buffer.sample_batch(
            self.batch_size,
            concatenate_samples,
            sample_method="random_unique",
            sample_attrs=["state", "action", "reward", "next_state", "terminal", "*"],
        )
        self.qnet.train()
        if self.mode == "vanilla":
            # Vanilla DQN, directly optimize q network.
            # target network is the same as the main network
            q_value = self._criticize(state)

            # gather requires long tensor, int32 is not accepted
            action_value = q_value.gather(
                dim=1,
                index=self.action_get_function(action).to(
                    device=q_value.device, dtype=t.long
                ),
            )

            target_next_q_value = (
                t.max(self._criticize(next_state), dim=1)[0].unsqueeze(1).detach()
            )
            y_i = self.reward_function(
                reward, self.discount, target_next_q_value, terminal, others
            )
            value_loss = self.criterion(action_value, y_i.type_as(action_value))

            if self.visualize:
                self.visualize_model(value_loss, "qnet", self.visualize_dir)

            if update_value:
                self.qnet.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.qnet.parameters(), self.grad_max)
                self.qnet_optim.step()

        elif self.mode == "fixed_target":
            # Fixed target DQN, which estimate next value by using the
            # target Q network. Similar to the idea of DDPG.
            q_value = self._criticize(state)

            # gather requires long tensor, int32 is not accepted
            action_value = q_value.gather(
                dim=1,
                index=self.action_get_function(action).to(
                    device=q_value.device, dtype=t.long
                ),
            )

            target_next_q_value = (
                t.max(self._criticize(next_state, True), dim=1)[0].unsqueeze(1).detach()
            )

            y_i = self.reward_function(
                reward, self.discount, target_next_q_value, terminal, others
            )
            value_loss = self.criterion(action_value, y_i.type_as(action_value))

            if self.visualize:
                self.visualize_model(value_loss, "qnet", self.visualize_dir)

            if update_value:
                self.qnet.zero_grad()
                self._backward(value_loss)
                nn.utils.clip_grad_norm_(self.qnet.parameters(), self.grad_max)
                self.qnet_optim.step()

            # Update target Q network
            if update_target:
                soft_update(self.qnet_target, self.qnet, self.update_rate)

        elif self.mode == "double":
            # Double DQN. DDQN also use the target network to estimate the next
            # value, but instead of selecting the maximum Q(s,a), it uses
            # the online DQN network to select an action and return Q(s,a'), to
            # reduce the over estimation.
            q_value = self._criticize(state)

            # gather requires long tensor, int32 is not accepted
            action_value = q_value.gather(
                dim=1,
                index=self.action_get_function(action).to(
                    device=q_value.device, dtype=t.long
                ),
            )

            with t.no_grad():
                target_next_q_value = self._criticize(next_state, True)
                next_action = self._act_discrete(next_state).to(
                    device=q_value.device, dtype=t.long
                )
                target_next_q_value = target_next_q_value.gather(
                    dim=1, index=next_action
                )

            y_i = self.reward_function(
                reward, self.discount, target_next_q_value, terminal, others
            )
            value_loss = self.criterion(action_value, y_i.type_as(action_value))

            if self.visualize:
                self.visualize_model(value_loss, "qnet", self.visualize_dir)

            if update_value:
                self.qnet.zero_grad()
                self._backward(value_loss)
                nn.utils.clip_grad_norm_(self.qnet.parameters(), self.grad_max)
                self.qnet_optim.step()

            # Update target Q network
            if update_target:
                if self.update_rate is not None:
                    soft_update(self.qnet_target, self.qnet, self.update_rate)
                else:
                    self._update_counter += 1
                    if self._update_counter % self.update_steps == 0:
                        hard_update(self.qnet_target, self.qnet)

        else:
            raise ValueError(f"Unknown DQN mode: {self.mode}")

        self.qnet.eval()
        # use .item() to prevent memory leakage
        return value_loss.item()

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "qnet_lr_sch"):
            self.qnet_lr_sch.step()

    def load(self, model_dir, network_map=None, version=-1):
        # DOC INHERITED
        super().load(model_dir, network_map, version)
        with t.no_grad():
            hard_update(self.qnet, self.qnet_target)

    @staticmethod
    def action_get_function(sampled_actions):
        """
        This function is used to get action numbers (int tensor indicating
        which discrete actions are used) from the sampled action dictionary.
        """
        return sampled_actions["action"]

    @staticmethod
    def reward_function(reward, discount, next_value, terminal, _):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * ~terminal * next_value

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        default_values = {
            "models": ["QNet", "QNet"],
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
            "epsilon_decay": 0.9999,
            "update_rate": 0.005,
            "update_steps": None,
            "learning_rate": 0.001,
            "discount": 0.99,
            "gradient_max": np.inf,
            "replay_size": 500000,
            "replay_device": "cpu",
            "replay_buffer": None,
            "mode": "double",
            "visualize": False,
            "visualize_dir": "",
        }
        config = deepcopy(config)
        config["frame"] = "DQN"
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
