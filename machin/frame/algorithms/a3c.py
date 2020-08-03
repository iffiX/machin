from .a2c import *
from machin.parallel.server import PushPullGradServer
from torch.optim import Adam


class A3C(A2C):
    """
    A3C framework.
    """
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 criterion: Callable,
                 grad_server: Tuple[PushPullGradServer,
                                    PushPullGradServer],
                 *_,
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

        Note:
            A3C algorithm relies on parameter servers to synchronize
            parameters of actor and critic models across samplers (
            interact with environment) and trainers (using samples
            to train.

            The parameter server type :class:`.PushPullGradServer`
            used here utilizes gradients calculated by trainers:

            1. perform a "sum" reduction process on the collected
            gradients, then apply this reduced gradient to the model
            managed by its primary reducer

            2. push the parameters of this updated managed model to
            a ordered key-value server so that all processes,
            including samplers and trainers, can access the updated
            parameters.

            The ``grad_servers`` argument is a pair of accessors to
            two :class:`.PushPullGradServerImpl` class.

        Args:
            actor: Actor network module.
            critic: Critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            grad_server: Custom gradient sync server accessors, the first
                server accessor is for actor, and the second one is for critic.
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
        # Adam is just a placeholder here, the actual optimizer is
        # set in parameter servers
        super(A3C, self).__init__(actor, critic, Adam, criterion,
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
        # disable local stepping
        self.actor_optim.step = lambda: None
        self.critic_optim.step = lambda: None
        self.actor_grad_server, self.critic_grad_server = \
            grad_server[0], grad_server[1]
        self.is_syncing = True

    def set_sync(self, is_syncing):
        self.is_syncing = is_syncing

    def manual_sync(self):
        self.actor_grad_server.pull(self.actor)
        self.critic_grad_server.pull(self.critic)

    def act(self, state: Dict[str, Any], **__):
        # DOC INHERITED
        if self.is_syncing:
            self.actor_grad_server.pull(self.actor)
        return super(A3C, self).act(state)

    def _eval_act(self,
                  state: Dict[str, Any],
                  action: Dict[str, Any],
                  **__):
        # DOC INHERITED
        if self.is_syncing:
            self.actor_grad_server.pull(self.actor)
        return super(A3C, self)._eval_act(state, action)

    def _criticize(self, state: Dict[str, Any], *_, **__):
        # DOC INHERITED
        if self.is_syncing:
            self.critic_grad_server.pull(self.critic)
        return super(A3C, self)._criticize(state)

    def update(self,
               update_value=True,
               update_policy=True,
               concatenate_samples=True,
               **__):
        # DOC INHERITED
        org_sync = self.is_syncing
        self.is_syncing = False
        super(A3C, self).update(update_value, update_policy,
                                concatenate_samples)
        self.is_syncing = org_sync
        self.actor_grad_server.push(self.actor)
        self.critic_grad_server.push(self.critic)
