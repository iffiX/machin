from .dqn_per import *
from .ddpg_per import *
from ..buffers.prioritized_buffer_d import DistributedPrioritizedBuffer
from machin.parallel.server import PushPullModelServer
from machin.parallel.distributed import RpcGroup


class DQNApex(DQNPer):
    """
    Massively parallel version of a Double DQN with prioritized replay.

    The pull function is invoked before using ``act_discrete``,
    ``act_discrete_with_noise`` and ``criticize``.

    The push function is invoked after ``update``.
    """
    def __init__(self,
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
                 update_rate: float = 0.005,
                 learning_rate: float = 0.001,
                 discount: float = 0.99,
                 replay_size: int = 500000,
                 **__):
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
            update_rate: :math:`\\tau` used to update target networks.
                Target parameters are updated as:

                :math:`\\theta_t = \\theta * \\tau + \\theta_t * (1 - \\tau)`
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            discount: :math:`\\gamma` used in the bellman function.
            replay_size: Local replay buffer size of a single worker.
        """
        super(DQNApex, self).__init__(qnet, qnet_target, optimizer, criterion,
                                      lr_scheduler=lr_scheduler,
                                      lr_scheduler_args=lr_scheduler_args,
                                      lr_scheduler_kwargs=lr_scheduler_kwargs,
                                      batch_size=batch_size,
                                      update_rate=update_rate,
                                      learning_rate=learning_rate,
                                      discount=discount)

        # will not support sharing rpc group,
        # use static buffer_name is ok here.
        self.replay_buffer = DistributedPrioritizedBuffer(
            buffer_name="buffer", group=apex_group,
            buffer_size=replay_size
        )
        self.apex_group = apex_group
        self.qnet_model_server = model_server[0]
        self.is_syncing = True

    def set_sync(self, is_syncing):
        self.is_syncing = is_syncing

    def manual_sync(self):
        self.qnet_model_server.pull(self.qnet)

    def act_discrete(self,
                     state: Dict[str, Any],
                     use_target: bool = False,
                     **__):
        # DOC INHERITED
        if self.is_syncing and not use_target:
            self.qnet_model_server.pull(self.qnet)
        return super(DQNApex, self).act_discrete(state, use_target)

    def act_discrete_with_noise(self,
                                state: Dict[str, Any],
                                use_target: bool = False,
                                **__):
        # DOC INHERITED
        if self.is_syncing and not use_target:
            self.qnet_model_server.pull(self.qnet)
        return super(DQNApex, self).act_discrete_with_noise(state, use_target)

    def update(self,
               update_value=True,
               update_target=True,
               concatenate_samples=True,
               **__):
        # DOC INHERITED
        result = super(DQNApex, self).update(update_value, update_target,
                                             concatenate_samples)
        if isinstance(self.qnet,
                      (nn.parallel.DataParallel,
                       nn.parallel.DistributedDataParallel)):
            self.qnet_model_server.push(self.qnet.module,
                                        pull_on_fail=False)
        else:
            self.qnet_model_server.push(self.qnet)
        return result


class DDPGApex(DDPGPer):
    """
    Massively parallel version of a DDPG with prioritized replay.

    The pull function is invoked before using
    ``act``, ``act_with_noise``, ``act_discrete``,
    ``act_discrete_with_noise`` and ``criticize``.

    The push function is invoked after ``update``.
    """
    def __init__(self,
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
                 learning_rate: float = 0.001,
                 discount: float = 0.99,
                 replay_size: int = 500000,
                 **__):
        """
        See Also:
            :class:`.DDPGPer`

        TODO:
            implement truncated n-step returns, just like the one used in
            :class:`.RAINBOW`.

        Hint:
            Your push and pull function will be called like::

                function(actor_model, "actor")

            The default implementation of pull and push functions
            is provided by :class:`.PushPullModelServer`.

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
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            discount: :math:`\\gamma` used in the bellman function.
            replay_size: Local replay buffer size of a single worker.
        """
        super(DDPGApex, self).__init__(actor, actor_target,
                                       critic, critic_target,
                                       optimizer, criterion,
                                       lr_scheduler=lr_scheduler,
                                       lr_scheduler_args=lr_scheduler_args,
                                       lr_scheduler_kwargs=lr_scheduler_kwargs,
                                       batch_size=batch_size,
                                       update_rate=update_rate,
                                       learning_rate=learning_rate,
                                       discount=discount)

        # will not support sharing rpc group,
        # use static buffer_name is ok here.
        self.replay_buffer = DistributedPrioritizedBuffer(
            buffer_name="buffer", group=apex_group,
            buffer_size=replay_size
        )
        self.apex_group = apex_group
        self.actor_model_server = model_server[0]
        self.is_syncing = True

    def set_sync(self, is_syncing):
        self.is_syncing = is_syncing

    def manual_sync(self):
        self.actor_model_server.pull(self.actor)

    def act(self,
            state: Dict[str, Any],
            use_target: bool = False,
            **__):
        # DOC INHERITED
        if self.is_syncing and not use_target:
            self.actor_model_server.pull(self.actor)
        return super(DDPGApex, self).act(state, use_target)

    def act_with_noise(self,
                       state: Dict[str, Any],
                       noise_param: Tuple = (0.0, 1.0),
                       ratio: float = 1.0,
                       mode: str = "uniform",
                       use_target: bool = False,
                       **__):
        # DOC INHERITED
        if self.is_syncing and not use_target:
            self.actor_model_server.pull(self.actor)
        return super(DDPGApex, self).act_with_noise(state,
                                                    noise_param=noise_param,
                                                    ratio=ratio,
                                                    mode=mode,
                                                    use_target=use_target)

    def act_discrete(self,
                     state: Dict[str, Any],
                     use_target: bool = False,
                     **__):
        # DOC INHERITED
        if self.is_syncing and not use_target:
            self.actor_model_server.pull(self.actor)
        return super(DDPGApex, self).act_discrete(state, use_target)

    def act_discrete_with_noise(self,
                                state: Dict[str, Any],
                                use_target: bool = False,
                                **__):
        # DOC INHERITED
        if self.is_syncing and not use_target:
            self.actor_model_server.pull(self.actor)
        return super(DDPGApex, self).act_discrete_with_noise(state, use_target)

    def update(self,
               update_value=True,
               update_policy=True,
               update_target=True,
               concatenate_samples=True,
               **__):
        # DOC INHERITED
        result = super(DDPGApex, self).update(update_value, update_policy,
                                              update_target,
                                              concatenate_samples)
        if isinstance(self.actor,
                      (nn.parallel.DataParallel,
                       nn.parallel.DistributedDataParallel)):
            self.actor_model_server.push(self.actor.module,
                                         pull_on_fail=False)
        else:
            self.actor_model_server.push(self.actor)
        return result
