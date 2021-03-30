# pylint: disable=wildcard-import, unused-wildcard-import
from .ddpg import *


class TD3(DDPG):
    """
    TD3 framework. Which adds a additional pair of critic and target critic
    network to DDPG.
    """

    _is_top = [
        "actor",
        "critic",
        "critic2",
        "actor_target",
        "critic_target",
        "critic2_target",
    ]
    _is_restorable = ["actor_target", "critic_target", "critic2_target"]

    def __init__(
        self,
        actor: Union[NeuralNetworkModule, nn.Module],
        actor_target: Union[NeuralNetworkModule, nn.Module],
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
        See Also:
            :class:`.DDPG`

        Args:
            actor: Actor network module.
            actor_target: Target actor network module.
            critic: Critic network module.
            critic_target: Target critic network module.
            critic2: The second critic network module.
            critic2_target: The second target critic network module.
            optimizer: Optimizer used to optimize ``actor``, ``critic``,
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
        if lr_scheduler_args is None:
            lr_scheduler_args = ((), (), ())
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = ({}, {}, {})

        super().__init__(
            actor,
            actor_target,
            critic,
            critic_target,
            optimizer,
            criterion,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=(
                lr_scheduler_args[:2] if lr_scheduler_args is not None else None
            ),
            lr_scheduler_kwargs=(
                lr_scheduler_kwargs[:2] if lr_scheduler_kwargs is not None else None
            ),
            batch_size=batch_size,
            update_rate=update_rate,
            update_steps=update_steps,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            discount=discount,
            gradient_max=gradient_max,
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=replay_buffer,
            visualize=visualize,
            visualize_dir=visualize_dir,
        )
        self.critic2 = critic2
        self.critic2_target = critic2_target
        self.critic2_optim = optimizer(
            self.critic2.parameters(), lr=critic_learning_rate
        )

        # Make sure target and online networks have the same weight
        with t.no_grad():
            hard_update(self.critic2, self.critic2_target)

        if lr_scheduler is not None:
            self.critic2_lr_sch = lr_scheduler(
                self.critic2_optim, *lr_scheduler_args[2], **lr_scheduler_kwargs[2]
            )

    @property
    def optimizers(self):
        return [self.actor_optim, self.critic_optim, self.critic2_optim]

    @optimizers.setter
    def optimizers(self, optimizers):
        self.actor_optim, self.critic_optim, self.critic2_optim = optimizers

    @property
    def lr_schedulers(self):
        if (
            hasattr(self, "actor_lr_sch")
            and hasattr(self, "critic_lr_sch")
            and hasattr(self, "critic2_lr_sch")
        ):
            return [self.actor_lr_sch, self.critic_lr_sch, self.critic2_lr_sch]
        return []

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

    def update(
        self,
        update_value=True,
        update_policy=True,
        update_target=True,
        concatenate_samples=True,
        **__
    ):
        # DOC INHERITED
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

        # Update critic network first.
        # Generate value reference :math: `y_i` using target actor and
        # target critic.
        with t.no_grad():
            next_action = self.action_transform_function(
                self.policy_noise_function(self._act(next_state, True)),
                next_state,
                others,
            )
            next_value = self._criticize(next_state, next_action, True)
            next_value2 = self._criticize2(next_state, next_action, True)
            next_value = t.min(next_value, next_value2)
            next_value = next_value.view(batch_size, -1)
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
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_max)
            self.critic_optim.step()
            self.critic2.zero_grad()
            value_loss2.backward()
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.gradient_max)
            self.critic2_optim.step()

        # Update actor network
        cur_action = self.action_transform_function(self._act(state), state, others)
        act_value = self._criticize(state, cur_action)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        if self.visualize:
            self.visualize_model(act_policy_loss, "actor", self.visualize_dir)

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_max)
            self.actor_optim.step()

        # Update target networks
        if update_target:
            if self.update_rate is not None:
                soft_update(self.actor_target, self.actor, self.update_rate)
                soft_update(self.critic_target, self.critic, self.update_rate)
                soft_update(self.critic2_target, self.critic2, self.update_rate)
            else:
                self._update_counter += 1
                if self._update_counter % self.update_steps == 0:
                    hard_update(self.actor_target, self.actor)
                    hard_update(self.critic_target, self.critic)
                    hard_update(self.critic2_target, self.critic2)

        self.actor.eval()
        self.critic.eval()
        self.critic2.eval()
        # use .item() to prevent memory leakage
        return (-act_policy_loss.item(), (value_loss.item() + value_loss2.item()) / 2)

    @staticmethod
    def policy_noise_function(actions, *_):
        # Function used to add noise to actions, mentioned in TD3
        # training tricks
        return actions

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "critic2_lr_sch"):
            self.critic2_lr_sch.step()
        super().update_lr_scheduler()

    def load(
        self, model_dir: str, network_map: Dict[str, str] = None, version: int = -1
    ):
        # DOC INHERITED
        TorchFramework.load(self, model_dir, network_map, version)
        with t.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic2, self.critic2_target)

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        config = DDPG.generate_config(config)
        config["frame"] = "TD3"
        config["frame_config"]["models"] = [
            "Actor",
            "Actor",
            "Critic",
            "Critic",
            "Critic",
            "Critic",
        ]
        config["frame_config"]["model_args"] = ((), (), (), (), (), ())
        config["frame_config"]["model_kwargs"] = ({}, {}, {}, {}, {}, {})
        return config
