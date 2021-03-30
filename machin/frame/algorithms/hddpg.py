# pylint: disable=wildcard-import, unused-wildcard-import
from .ddpg import *


class HDDPG(DDPG):
    """
    HDDPG framework.
    """

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
        update_rate: float = 0.005,
        update_steps: Union[int, None] = None,
        actor_learning_rate: float = 0.0005,
        critic_learning_rate: float = 0.001,
        discount: float = 0.99,
        gradient_max: float = np.inf,
        q_increase_rate: float = 1.0,
        q_decrease_rate: float = 1.0,
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
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=replay_buffer,
            visualize=visualize,
            visualize_dir=visualize_dir,
        )
        self.q_increase_rate = q_increase_rate
        self.q_decrease_rate = q_decrease_rate

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
        value_diff = y_i.type_as(cur_value) - cur_value
        value_change = t.where(
            value_diff > 0,
            value_diff * self.q_increase_rate,
            value_diff * self.q_decrease_rate,
        )
        value_loss = self.criterion(cur_value, (cur_value + value_change).detach())

        if self.visualize:
            self.visualize_model(value_loss, "critic", self.visualize_dir)

        if update_value:
            self.critic.zero_grad()
            self._backward(value_loss)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_max)
            self.critic_optim.step()

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

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        config = DDPG.generate_config(config)
        config["frame"] = "HDDPG"
        config["frame_config"]["q_increase_rate"] = 1.0
        config["frame_config"]["q_decrease_rate"] = 1.0
        return config
