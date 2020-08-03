from .a2c import *


class PPO(A2C):
    """
    PPO framework.
    """
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 criterion: Callable,
                 *_,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple] = (),
                 lr_scheduler_kwargs: Tuple[Dict, Dict] = (),
                 actor_learning_rate: float = 0.001,
                 critic_learning_rate: float = 0.001,
                 entropy_weight: float = None,
                 value_weight: float = 0.5,
                 surrogate_loss_clip: float = 0.2,
                 gradient_max: float = np.inf,
                 gae_lambda: float = 1.0,
                 discount: float = 0.99,
                 update_times: int = 50,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 visualize: bool = False,
                 visualize_dir: str = "",
                 **__):
        """
        See Also:
            :class:`.A2C`

        Args:
            actor: Actor network module.
            critic: Critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            actor_learning_rate: Learning rate of the actor optimizer,
                not compatible with ``lr_scheduler``.
            critic_learning_rate: Learning rate of the critic optimizer,
                not compatible with ``lr_scheduler``.
            entropy_weight: Weight of entropy in your loss function, a positive
                entropy weight will minimize entropy, while a negative one will
                maximize entropy.
            value_weight: Weight of critic value loss.
            surrogate_loss_clip: Surrogate loss clipping parameter in PPO.
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
            visualize_dir: Visualized graph save directory.
        """
        super(PPO, self).__init__(actor, critic, optimizer, criterion,
                                  lr_scheduler=lr_scheduler,
                                  lr_scheduler_args=lr_scheduler_args,
                                  lr_scheduler_kwargs=lr_scheduler_kwargs,
                                  actor_learning_rate=actor_learning_rate,
                                  critic_learning_rate=critic_learning_rate,
                                  entropy_weight=entropy_weight,
                                  value_weight=value_weight,
                                  gradient_max=gradient_max,
                                  gae_lambda=gae_lambda,
                                  discount=discount,
                                  replay_size=replay_size,
                                  replay_device=replay_device,
                                  replay_buffer=replay_buffer,
                                  visualize=visualize,
                                  visualize_dir=visualize_dir)
        self.update_times = update_times
        self.surr_clip = surrogate_loss_clip

    def update(self,
               update_value=True,
               update_policy=True,
               concatenate_samples=True,
               **__):
        # DOC INHERITED
        sum_act_policy_loss = 0
        sum_value_loss = 0

        # sample a batch
        batch_size, (state, action, reward, next_state,
                     terminal, target_value, advantage) = \
            self.replay_buffer.sample_batch(-1,
                                            sample_method="all",
                                            concatenate=concatenate_samples,
                                            sample_attrs=[
                                                "state", "action", "reward",
                                                "next_state", "terminal",
                                                "value", "gae"],
                                            additional_concat_attrs=[
                                                "value", "gae"
                                            ])

        # normalize advantage
        advantage = ((advantage - advantage.mean()) /
                     (advantage.std() + 1e-6))

        # Infer original action log probability
        __, action_log_prob, *_ = self._eval_act(state, action)
        action_log_prob = action_log_prob.view(batch_size, 1).detach()

        for _ in range(self.update_times):
            if self.entropy_weight is not None:
                __, new_action_log_prob, new_action_entropy, *_ = \
                    self._eval_act(state, action)
            else:
                __, new_action_log_prob, *_ = \
                    self._eval_act(state, action)
                new_action_entropy = None

            new_action_log_prob = new_action_log_prob.view(batch_size, 1)

            # calculate surrogate loss
            # The function of this process is ignoring actions that are not
            # likely to be produced in current actor policy distribution,
            # Because in each update, the old policy distribution diverges
            # from the current distribution more and more.
            sim_ratio = t.exp(new_action_log_prob - action_log_prob)
            advantage = advantage.to(sim_ratio.device)
            surr_loss_1 = sim_ratio * advantage
            surr_loss_2 = t.clamp(sim_ratio,
                                  1 - self.surr_clip,
                                  1 + self.surr_clip) * advantage

            # calculate policy loss using surrogate loss
            act_policy_loss = -t.min(surr_loss_1, surr_loss_2)

            if new_action_entropy is not None:
                act_policy_loss += (self.entropy_weight *
                                    new_action_entropy.mean())

            act_policy_loss = act_policy_loss.mean()

            if self.visualize:
                self.visualize_model(act_policy_loss, "actor",
                                     self.visualize_dir)

            # Update actor network
            if update_policy:
                self.actor.zero_grad()
                act_policy_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.grad_max
                )
                self.actor_optim.step()
            sum_act_policy_loss += act_policy_loss.item()

            # calculate value loss
            value = self._criticize(state)
            value_loss = (self.criterion(target_value.to(value.device),
                                         value) *
                          self.value_weight)

            if self.visualize:
                self.visualize_model(value_loss, "critic",
                                     self.visualize_dir)

            # Update critic network
            if update_value:
                self.critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.grad_max
                )
                self.critic_optim.step()
            sum_value_loss += value_loss.item()

        self.replay_buffer.clear()
        return (-sum_act_policy_loss / self.update_times,
                sum_value_loss / self.update_times)
