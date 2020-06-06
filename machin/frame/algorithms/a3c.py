from .a2c import *
from machin.parallel.server import PushPullGradServer
from machin.parallel.distributed import RpcGroup
from machin.utils.helper_classes import Switch


class A3C(A2C):
    """
    A3C framework.
    """
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 criterion: Callable,
                 *_,
                 grad_server_group: RpcGroup = None,
                 grad_servers: Tuple[Any, Any] = None,
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
        See Also:
            :class:`.A2C`

        Args:
            actor: Actor network module.
            critic: Critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            grad_server_group: Gradient sync server group, actor and critic will
                share the same sync group.
            grad_servers: Custom gradient sync servers, the first server is for
                actor, and the second one is for critic.
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
        super(A3C, self).__init__(actor, critic, optimizer, criterion,
                                  lr_scheduler=lr_scheduler,
                                  lr_scheduler_args=lr_scheduler_args,
                                  lr_scheduler_kwargs=lr_scheduler_kwargs,
                                  learning_rate=learning_rate,
                                  entropy_weight=entropy_weight,
                                  value_weight=value_weight,
                                  gradient_max=gradient_max,
                                  gae_lambda=gae_lambda,
                                  discount=discount,
                                  update_times=update_times,
                                  replay_size=replay_size,
                                  replay_device=replay_device,
                                  replay_buffer=replay_buffer,
                                  visualize=visualize)
        if grad_servers is None:
            self.actor_grad_server = PushPullGradServer(
                group=grad_server_group
            )
            self.critic_grad_server = PushPullGradServer(
                group=grad_server_group
            )
        else:
            self.actor_grad_server, self.critic_grad_server = \
                grad_servers[0], grad_servers[1]
        self._disable_sync = Switch()

    def act(self, state: Dict[str, Any], pull: bool = True, **__):
        # DOC INHERITED
        if pull and not self._disable_sync.get():
            self.actor_grad_server.pull(self.actor)
        return safe_call(self.actor, state)

    def eval_act(self,
                 state: Dict[str, Any],
                 action: Dict[str, Any],
                 pull: bool = True,
                 **__):
        # DOC INHERITED
        if pull and not self._disable_sync.get():
            self.actor_grad_server.pull(self.actor)
        return safe_call(self.actor, state, action)

    def criticize(self, state: Dict[str, Any], *_, pull=True, **__):
        # DOC INHERITED
        if pull and not self._disable_sync.get():
            self.critic_grad_server.pull(self.critic)
        return safe_call(self.critic, state)

    def update(self,
               update_value=True,
               update_policy=True,
               concatenate_samples=True,
               **__):
        # DOC INHERITED
        self._disable_sync.on()
        self.update(update_value, update_policy, concatenate_samples)
        self._disable_sync.off()
        self.actor_grad_server.push(self.actor)
        self.critic_grad_server.push(self.critic)
