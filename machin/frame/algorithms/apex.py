from .dqn_per import *
from .ddpg_per import *
from ..buffers.prioritized_buffer_d import DistributedPrioritizedBuffer
from torch.nn.parallel import DistributedDataParallel
from machin.parallel.server import PushPullModelServer
from machin.parallel.distributed import get_world, RpcGroup
from machin.frame.helpers.servers import model_server_helper


def _disable_update(*_, **__):
    return None, None


class DQNApex(DQNPer):
    """
    Massively parallel version of a Double DQN with prioritized replay.

    The pull function is invoked before using ``act_discrete``,
    ``act_discrete_with_noise`` and ``criticize``.

    The push function is invoked after ``update``.
    """

    def __init__(
        self,
        qnet: Union[NeuralNetworkModule, nn.Module],
        qnet_target: Union[NeuralNetworkModule, nn.Module],
        optimizer: Callable,
        criterion: Callable,
        apex_group: RpcGroup,
        model_server: Tuple[PushPullModelServer],
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple] = (),
        lr_scheduler_kwargs: Tuple[Dict] = (),
        batch_size: int = 100,
        epsilon_decay: float = 0.9999,
        update_rate: float = 0.005,
        update_steps: Union[int, None] = None,
        learning_rate: float = 0.001,
        discount: float = 0.99,
        gradient_max: float = np.inf,
        replay_size: int = 500000,
        **__,
    ):
        """
        See Also:
            :class:`.DQNPer`

        Note:
            Apex framework supports multiple workers(samplers), and only
            one trainer, you may use ``DistributedDataParallel`` in trainer.
            If you use ``DistributedDataParallel``, you must call ``update()``
            in all member processes of ``DistributedDataParallel``.

        Args:
            qnet: Q network module.
            qnet_target: Target Q network module.
            optimizer: Optimizer used to optimize ``qnet``.
            criterion: Criterion used to evaluate the value loss.
            apex_group: Group of all processes using the apex-DQN framework,
                including all samplers and trainers.
            model_server: Custom model sync server accessor for ``qnet``.
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
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            discount: :math:`\\gamma` used in the bellman function.
            gradient_max: Maximum gradient.
            replay_size: Local replay buffer size of a single worker.
        """
        super().__init__(
            qnet,
            qnet_target,
            optimizer,
            criterion,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            batch_size=batch_size,
            epsilon_decay=epsilon_decay,
            update_rate=update_rate,
            update_steps=update_steps,
            learning_rate=learning_rate,
            discount=discount,
            gradient_max=gradient_max,
        )
        self._is_using_DP_or_DDP = isinstance(
            self.qnet, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
        )

        # will not support sharing rpc group,
        # use static buffer_name is ok here.
        self.replay_buffer = DistributedPrioritizedBuffer(
            buffer_name="buffer", group=apex_group, buffer_size=replay_size
        )
        self.apex_group = apex_group
        self.qnet_model_server = model_server[0]
        self.is_syncing = True

    @classmethod
    def is_distributed(cls):
        return True

    def set_sync(self, is_syncing):
        self.is_syncing = is_syncing

    def manual_sync(self):
        if not self._is_using_DP_or_DDP:
            self.qnet_model_server.pull(self.qnet)

    def act_discrete(self, state: Dict[str, Any], use_target: bool = False, **__):
        # DOC INHERITED
        if self.is_syncing and not use_target and not self._is_using_DP_or_DDP:
            self.qnet_model_server.pull(self.qnet)
        return super().act_discrete(state, use_target)

    def act_discrete_with_noise(
        self,
        state: Dict[str, Any],
        use_target: bool = False,
        decay_epsilon: bool = True,
        **__,
    ):
        # DOC INHERITED
        if self.is_syncing and not use_target and not self._is_using_DP_or_DDP:
            self.qnet_model_server.pull(self.qnet)
        return super().act_discrete_with_noise(state, use_target, decay_epsilon)

    def update(
        self, update_value=True, update_target=True, concatenate_samples=True, **__
    ):
        # DOC INHERITED
        result = super().update(update_value, update_target, concatenate_samples)
        if self._is_using_DP_or_DDP:
            self.qnet_model_server.push(self.qnet.module, pull_on_fail=False)
        else:
            self.qnet_model_server.push(self.qnet)
        return result

    @classmethod
    def generate_config(cls, config: Dict[str, Any]):
        default_values = {
            "learner_process_number": 1,
            "model_server_group_name": "dqn_apex_model_server",
            "model_server_members": "all",
            "apex_group_name": "dqn_apex",
            "apex_members": "all",
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
        }
        config = deepcopy(config)
        config["frame"] = "DQNApex"
        config["batch_num"] = {"sampler": 10, "learner": 1}
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
        world = get_world()
        f_config = deepcopy(config["frame_config"])
        apex_group = world.create_rpc_group(
            group_name=f_config["apex_group_name"],
            members=(
                world.get_members()
                if f_config["apex_members"] == "all"
                else f_config["apex_members"]
            ),
        )

        models = assert_and_get_valid_models(f_config["models"])
        model_args = f_config["model_args"]
        model_kwargs = f_config["model_kwargs"]
        models = [
            m(*arg, **kwarg).to(model_device)
            for m, arg, kwarg in zip(models, model_args, model_kwargs)
        ]
        # wrap models in DistributedDataParallel when running in learner mode
        max_learner_id = f_config["learner_process_number"]

        learner_group = world.create_collective_group(ranks=list(range(max_learner_id)))

        if world.rank < max_learner_id:
            models = [
                DistributedDataParallel(module=m, process_group=learner_group.group)
                for m in models
            ]

        optimizer = assert_and_get_valid_optimizer(f_config["optimizer"])
        criterion = assert_and_get_valid_criterion(f_config["criterion"])(
            *f_config["criterion_args"], **f_config["criterion_kwargs"]
        )
        criterion.reduction = "none"
        lr_scheduler = f_config["lr_scheduler"] and assert_and_get_valid_lr_scheduler(
            f_config["lr_scheduler"]
        )
        servers = model_server_helper(
            model_num=1,
            group_name=f_config["model_server_group_name"],
            members=f_config["model_server_members"],
        )
        del f_config["optimizer"]
        del f_config["criterion"]
        del f_config["lr_scheduler"]
        frame = cls(
            *models,
            optimizer,
            criterion,
            apex_group,
            servers,
            lr_scheduler=lr_scheduler,
            **f_config,
        )
        if world.rank >= max_learner_id:
            frame.role = "sampler"
            frame.update = _disable_update
        else:
            frame.role = "learner"
        return frame


class DDPGApex(DDPGPer):
    """
    Massively parallel version of a DDPG with prioritized replay.

    The pull function is invoked before using
    ``act``, ``act_with_noise``, ``act_discrete``,
    ``act_discrete_with_noise`` and ``criticize``.

    The push function is invoked after ``update``.
    """

    def __init__(
        self,
        actor: Union[NeuralNetworkModule, nn.Module],
        actor_target: Union[NeuralNetworkModule, nn.Module],
        critic: Union[NeuralNetworkModule, nn.Module],
        critic_target: Union[NeuralNetworkModule, nn.Module],
        optimizer: Callable,
        criterion: Callable,
        apex_group: RpcGroup,
        model_server: Tuple[PushPullModelServer],
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple, Tuple] = (),
        lr_scheduler_kwargs: Tuple[Dict, Dict] = (),
        batch_size: int = 100,
        update_rate: float = 0.005,
        update_steps: Union[int, None] = None,
        actor_learning_rate: float = 0.0005,
        critic_learning_rate: float = 0.001,
        discount: float = 0.99,
        gradient_max: float = np.inf,
        replay_size: int = 500000,
        **__,
    ):
        """
        See Also:
            :class:`.DDPGPer`

        TODO:
            implement truncated n-step returns, just like the one used in
            :class:`.RAINBOW`.

        Note:
            Apex framework supports multiple workers(samplers), and only
            one trainer, you may use ``DistributedDataParallel`` in trainer.
            If you use ``DistributedDataParallel``, you must call ``update()``
            in all member processes of ``DistributedDataParallel``.

        Args:
            actor: Actor network module.
            actor_target: Target actor network module.
            critic: Critic network module.
            critic_target: Target critic network module.
            optimizer: Optimizer used to optimize ``qnet``.
            criterion: Criterion used to evaluate the value loss.
            apex_group: Group of all processes using the apex-DDPG framework,
                including all samplers and trainers.
            model_server: Custom model sync server accessor for ``actor``.
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
            gradient_max: Maximum gradient.
            replay_size: Local replay buffer size of a single worker.
        """
        super().__init__(
            actor,
            actor_target,
            critic,
            critic_target,
            optimizer,
            criterion,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            batch_size=batch_size,
            update_rate=update_rate,
            update_steps=update_steps,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            discount=discount,
            gradient_max=gradient_max,
        )
        self._is_using_DP_or_DDP = isinstance(
            self.actor, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
        )
        # will not support sharing rpc group,
        # use static buffer_name is ok here.
        self.replay_buffer = DistributedPrioritizedBuffer(
            buffer_name="buffer", group=apex_group, buffer_size=replay_size
        )
        self.apex_group = apex_group
        self.actor_model_server = model_server[0]
        self.is_syncing = True

    @classmethod
    def is_distributed(cls):
        return True

    def set_sync(self, is_syncing):
        self.is_syncing = is_syncing

    def manual_sync(self):
        if not self._is_using_DP_or_DDP:
            self.actor_model_server.pull(self.actor)

    def act(self, state: Dict[str, Any], use_target: bool = False, **__):
        # DOC INHERITED
        if self.is_syncing and not use_target and not self._is_using_DP_or_DDP:
            self.actor_model_server.pull(self.actor)
        return super().act(state, use_target)

    def act_with_noise(
        self,
        state: Dict[str, Any],
        noise_param: Tuple = (0.0, 1.0),
        ratio: float = 1.0,
        mode: str = "uniform",
        use_target: bool = False,
        **__,
    ):
        # DOC INHERITED
        if self.is_syncing and not use_target and not self._is_using_DP_or_DDP:
            self.actor_model_server.pull(self.actor)
        return super().act_with_noise(
            state,
            noise_param=noise_param,
            ratio=ratio,
            mode=mode,
            use_target=use_target,
        )

    def act_discrete(self, state: Dict[str, Any], use_target: bool = False, **__):
        # DOC INHERITED
        if self.is_syncing and not use_target and not self._is_using_DP_or_DDP:
            self.actor_model_server.pull(self.actor)
        return super().act_discrete(state, use_target)

    def act_discrete_with_noise(
        self, state: Dict[str, Any], use_target: bool = False, **__
    ):
        # DOC INHERITED
        if self.is_syncing and not use_target and not self._is_using_DP_or_DDP:
            self.actor_model_server.pull(self.actor)
        return super().act_discrete_with_noise(state, use_target)

    def update(
        self,
        update_value=True,
        update_policy=True,
        update_target=True,
        concatenate_samples=True,
        **__,
    ):
        # DOC INHERITED
        result = super().update(
            update_value, update_policy, update_target, concatenate_samples
        )
        if self._is_using_DP_or_DDP:
            self.actor_model_server.push(self.actor.module, pull_on_fail=False)
        else:
            self.actor_model_server.push(self.actor)
        return result

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        default_values = {
            "learner_process_number": 1,
            "model_server_group_name": "ddpg_apex_model_server",
            "model_server_members": "all",
            "apex_group_name": "ddpg_apex",
            "apex_members": "all",
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
            "update_rate": 0.005,
            "update_steps": None,
            "learning_rate": 0.001,
            "discount": 0.99,
            "gradient_max": np.inf,
            "replay_size": 500000,
        }
        config = deepcopy(config)
        config["frame"] = "DDPGApex"
        config["batch_num"] = {"sampler": 10, "learner": 1}
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
        world = get_world()
        f_config = deepcopy(config["frame_config"])
        apex_group = world.create_rpc_group(
            group_name=f_config["apex_group_name"],
            members=(
                world.get_members()
                if f_config["apex_members"] == "all"
                else f_config["apex_members"]
            ),
        )

        models = assert_and_get_valid_models(f_config["models"])
        model_args = f_config["model_args"]
        model_kwargs = f_config["model_kwargs"]
        models = [
            m(*arg, **kwarg).to(model_device)
            for m, arg, kwarg in zip(models, model_args, model_kwargs)
        ]
        # wrap models in DistributedDataParallel when running in learner mode
        max_learner_id = f_config["learner_process_number"]

        learner_group = world.create_collective_group(ranks=list(range(max_learner_id)))

        if world.rank < max_learner_id:
            models = [
                DistributedDataParallel(module=m, process_group=learner_group.group)
                for m in models
            ]

        optimizer = assert_and_get_valid_optimizer(f_config["optimizer"])
        criterion = assert_and_get_valid_criterion(f_config["criterion"])(
            *f_config["criterion_args"], **f_config["criterion_kwargs"]
        )
        criterion.reduction = "none"
        lr_scheduler = f_config["lr_scheduler"] and assert_and_get_valid_lr_scheduler(
            f_config["lr_scheduler"]
        )
        servers = model_server_helper(
            model_num=1,
            group_name=f_config["model_server_group_name"],
            members=f_config["model_server_members"],
        )
        del f_config["optimizer"]
        del f_config["criterion"]
        del f_config["lr_scheduler"]
        frame = cls(
            *models,
            optimizer,
            criterion,
            apex_group,
            servers,
            lr_scheduler=lr_scheduler,
            **f_config,
        )
        if world.rank >= max_learner_id:
            frame.role = "sampler"
            frame.update = _disable_update
        else:
            frame.role = "learner"
        return frame
