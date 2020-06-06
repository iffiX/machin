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
                 learning_rate: float = 0.001,
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
                 **__):
        """
        See Also:
            :class:`.A2C`

        Args:
            actor: Actor network module.
            critic: Critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Critierion used to evaluate the value loss.
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
            surrogate_loss_clip: :math:`\\epsilon` used in surrogate loss
                clipping:
                :math:`L_t(\\theta)=min(r_t(\\theta)\\hat{A_t}, \
                                        clip(r_t(\\theta), \
                                             1-\\epsilon, 1+\\epsilon) \
                                        \\hat{A_t})`
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
        super(PPO, self).__init__(actor, critic, optimizer, criterion,
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
        batch_size, (state, action, reward, next_state, terminal,
                     action_log_prob, target_value, advantage, *others) = \
            self.replay_buffer.sample_batch(-1,
                                            sample_method="all",
                                            concatenate=concatenate_samples,
                                            sample_attrs=[
                                                "state", "action", "reward",
                                                "next_state", "terminal",
                                                "action_log_prob", "value",
                                                "gae", "*"],
                                            additional_concat_attrs=[
                                                "action_log_prob", "value",
                                                "gae"
                                            ])

        # normalize target value
        target_value = ((target_value - target_value.mean()) /
                        (target_value.std() + 1e-5))

        # Infer original action log probability
        __, action_log_prob, *_ = self.eval_act(state, action)
        action_log_prob = action_log_prob.view(batch_size, 1)

        for i in range(self.update_times):
            value = self.criticize(state)

            if self.entropy_weight is not None:
                __, new_action_log_prob, new_action_entropy, *_ = \
                    self.eval_act(state, action)

            else:
                __, new_action_log_prob, *_ = self.eval_act(state, action)
                new_action_entropy = None

            new_action_log_prob = new_action_log_prob.view(batch_size, 1)

            # calculate surrogate loss
            sim_ratio = t.exp(new_action_log_prob - action_log_prob).detach()
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

            value_loss = (self.criterion(target_value.to(value.device), value) *
                          self.value_weight)

            if self.visualize:
                self.visualize_model(act_policy_loss, "actor")

            # Update actor network
            if update_policy:
                self.actor.zero_grad()
                act_policy_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.grad_max
                )
                self.actor_optim.step()
                sum_act_policy_loss += act_policy_loss.item()

            if self.visualize:
                self.visualize_model(value_loss, "critic")

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
