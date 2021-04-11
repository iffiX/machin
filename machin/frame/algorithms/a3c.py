from .a2c import *
from machin.parallel.server import PushPullGradServer
from machin.frame.helpers.servers import grad_server_helper
from .utils import FakeOptimizer, assert_and_get_valid_lr_scheduler


class A3C(A2C):
    """
    A3C framework.
    """

    def __init__(
        self,
        actor: Union[NeuralNetworkModule, nn.Module],
        critic: Union[NeuralNetworkModule, nn.Module],
        criterion: Callable,
        grad_server: Tuple[PushPullGradServer, PushPullGradServer],
        *_,
        batch_size: int = 100,
        actor_update_times: int = 5,
        critic_update_times: int = 10,
        entropy_weight: float = None,
        value_weight: float = 0.5,
        gradient_max: float = np.inf,
        gae_lambda: float = 1.0,
        discount: float = 0.99,
        normalize_advantage: bool = True,
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
            criterion: Criterion used to evaluate the value loss.
            grad_server: Custom gradient sync server accessors, the first
                server accessor is for actor, and the second one is for critic.
            batch_size: Batch size used during training.
            actor_update_times: Times to update actor in ``update()``.
            critic_update_times: Times to update critic in ``update()``.
            entropy_weight: Weight of entropy in your loss function, a positive
                entropy weight will minimize entropy, while a negative one will
                maximize entropy.
            value_weight: Weight of critic value loss.
            gradient_max: Maximum gradient.
            gae_lambda: :math:`\\lambda` used in generalized advantage
                estimation.
            discount: :math:`\\gamma` used in the bellman function.
            normalize_advantage: Whether to normalize the advantage function.
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            visualize: Whether visualize the network flow in the first pass.
            visualize_dir: Visualized graph save directory.
        """
        # Adam is just a placeholder here, the actual optimizer is
        # set in parameter servers
        super().__init__(
            actor,
            critic,
            FakeOptimizer,
            criterion,
            batch_size=batch_size,
            actor_update_times=actor_update_times,
            critic_update_times=critic_update_times,
            entropy_weight=entropy_weight,
            value_weight=value_weight,
            gradient_max=gradient_max,
            gae_lambda=gae_lambda,
            discount=discount,
            normalize_advantage=normalize_advantage,
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=replay_buffer,
            visualize=visualize,
            visualize_dir=visualize_dir,
        )
        # disable local stepping
        self.actor_optim.step = lambda: None
        self.critic_optim.step = lambda: None
        self.actor_grad_server, self.critic_grad_server = grad_server[0], grad_server[1]
        self.is_syncing = True

    @property
    def optimizers(self):
        return []

    @optimizers.setter
    def optimizers(self, optimizers):
        pass

    @property
    def lr_schedulers(self):
        return []

    @classmethod
    def is_distributed(cls):
        return True

    def set_sync(self, is_syncing):
        self.is_syncing = is_syncing

    def manual_sync(self):
        self.actor_grad_server.pull(self.actor)
        self.critic_grad_server.pull(self.critic)

    def act(self, state: Dict[str, Any], **__):
        # DOC INHERITED
        if self.is_syncing:
            self.actor_grad_server.pull(self.actor)
        return super().act(state)

    def _eval_act(self, state: Dict[str, Any], action: Dict[str, Any], **__):
        # DOC INHERITED
        if self.is_syncing:
            self.actor_grad_server.pull(self.actor)
        return super()._eval_act(state, action)

    def _criticize(self, state: Dict[str, Any], *_, **__):
        # DOC INHERITED
        if self.is_syncing:
            self.critic_grad_server.pull(self.critic)
        return super()._criticize(state)

    def update(
        self, update_value=True, update_policy=True, concatenate_samples=True, **__
    ):
        # DOC INHERITED
        org_sync = self.is_syncing
        self.is_syncing = False
        super().update(update_value, update_policy, concatenate_samples)
        self.is_syncing = org_sync
        self.actor_grad_server.push(self.actor)
        self.critic_grad_server.push(self.critic)

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        default_values = {
            "grad_server_group_name": "a3c_grad_server",
            "grad_server_members": "all",
            "models": ["Actor", "Critic"],
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
            "actor_update_times": 5,
            "critic_update_times": 10,
            "actor_learning_rate": 0.001,
            "critic_learning_rate": 0.001,
            "entropy_weight": None,
            "value_weight": 0.5,
            "gradient_max": np.inf,
            "gae_lambda": 1.0,
            "discount": 0.99,
            "normalize_advantage": True,
            "replay_size": 500000,
            "replay_device": "cpu",
            "replay_buffer": None,
            "visualize": False,
            "visualize_dir": "",
        }
        config = deepcopy(config)
        config["frame"] = "A3C"
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
        model_cls = assert_and_get_valid_models(f_config["models"])
        model_args = f_config["model_args"]
        model_kwargs = f_config["model_kwargs"]
        models = [
            m(*arg, **kwarg).to(model_device)
            for m, arg, kwarg in zip(model_cls, model_args, model_kwargs)
        ]
        model_creators = [
            lambda: m(*arg, **kwarg)
            for m, arg, kwarg in zip(model_cls, model_args, model_kwargs)
        ]
        optimizer = assert_and_get_valid_optimizer(f_config["optimizer"])
        criterion = assert_and_get_valid_criterion(f_config["criterion"])(
            *f_config["criterion_args"], **f_config["criterion_kwargs"]
        )
        lr_scheduler = f_config["lr_scheduler"] and assert_and_get_valid_lr_scheduler(
            f_config["lr_scheduler"]
        )

        servers = grad_server_helper(
            model_creators,
            group_name=f_config["grad_server_group_name"],
            members=f_config["grad_server_members"],
            optimizer=optimizer,
            learning_rate=[
                f_config["actor_learning_rate"],
                f_config["critic_learning_rate"],
            ],
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=f_config["lr_scheduler_args"] or ((), ()),
            lr_scheduler_kwargs=f_config["lr_scheduler_kwargs"] or ({}, {}),
        )
        del f_config["criterion"]
        frame = cls(*models, criterion, servers, **f_config)
        return frame
