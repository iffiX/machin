from machin.utils.logging import default_logger
# pylint: disable=wildcard-import, unused-wildcard-import
from .ddpg import *


class TD3(DDPG):
    """
    TD3 framework. Which adds a additional pair of critic and target critic
    network to DDPG.
    """

    _is_top = ["actor", "critic", "critic2",
               "actor_target", "critic_target", "critic2_target"]
    _is_restorable = ["actor_target", "critic_target", "critic2_target"]

    def __init__(self,
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
                 lr_scheduler_args: Tuple[Tuple, Tuple, Tuple] = (),
                 lr_scheduler_kwargs: Tuple[Dict, Dict, Dict] = (),
                 batch_size: int = 100,
                 update_rate: float = 0.005,
                 learning_rate: float = 0.001,
                 discount: float = 0.99,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 policy_noise_func: Callable = None,
                 reward_func: Callable = None,
                 action_trans_func: Callable = None,
                 visualize: bool = False,
                 **__):
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
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used during training.
            update_rate: :math:`\\tau` used to update target networks.
                Target parameters are updated as:

                :math:`\\theta_t = \\theta * \\tau + \\theta_t * (1 - \\tau)`

            discount: :math:`\\gamma` used in the bellman function.
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            reward_func: Reward function used in training.
            action_trans_func: Action transform function, used to transform
                the raw output of your actor, by default it is:
                ``lambda act: {"action": act}``
            visualize: Whether visualize the network flow in the first pass.
        """
        super(TD3, self).__init__(
            actor, actor_target, critic, critic_target, optimizer, criterion,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args[:2],
            lr_scheduler_kwargs=lr_scheduler_kwargs[:2],
            batch_size=batch_size,
            update_rate=update_rate,
            learning_rate=learning_rate,
            discount=discount,
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=replay_buffer,
            reward_func=reward_func,
            action_trans_func=action_trans_func,
            visualize=visualize
        )
        self.critic2 = critic2
        self.critic2_target = critic2_target
        self.critic2_optim = optimizer(self.critic2.parameters(),
                                       lr=learning_rate)

        # Make sure target and online networks have the same weight
        with t.no_grad():
            hard_update(self.critic2, self.critic2_target)

        if lr_scheduler is not None:
            self.critic2_lr_sch = lr_scheduler(
                self.critic2_optim,
                *lr_scheduler_args[2],
                **lr_scheduler_kwargs[2]
            )

        if policy_noise_func is None:
            default_logger.warning("Policy noise function is None, "
                                   "no policy noise will be applied "
                                   "during update!")
        self.policy_noise_func = (TD3._policy_noise_function
                                  if policy_noise_func is None
                                  else policy_noise_func)

    def criticize2(self,
                   state: Dict[str, Any],
                   action: Dict[str, Any],
                   use_target=False,
                   **__):
        """
        Use the second critic network to evaluate current value.

        Args:
            state: Current state.
            action: Current action.
            use_target: Whether to use the target network.

        Returns:
            Value of shape ``[batch_size, 1]``.
        """
        if use_target:
            return safe_call(self.critic2_target, state, action)
        else:
            return safe_call(self.critic2, state, action)

    def update(self,
               update_value=True,
               update_policy=True,
               update_target=True,
               concatenate_samples=True,
               **__):
        # DOC INHERITED
        batch_size, (state, action, reward, next_state, terminal, *others) = \
            self.replay_buffer.sample_batch(self.batch_size,
                                            concatenate_samples,
                                            sample_attrs=[
                                                "state", "action",
                                                "reward", "next_state",
                                                "terminal", "*"
                                            ])

        # Update critic network first.
        # Generate value reference :math: `y_i` using target actor and
        # target critic.
        with t.no_grad():
            next_action = self.action_transform_func(
                self.policy_noise_func(
                    self.act(next_state, True)
                ),
                next_state, *others
            )
            next_value = self.criticize(next_state, next_action, True)
            next_value2 = self.criticize2(next_state, next_action, True)
            next_value = t.min(next_value, next_value2)
            next_value = next_value.view(batch_size, -1)
            y_i = self.reward_func(reward, self.discount, next_value, terminal,
                                   *others)

        cur_value = self.criticize(state, action)
        cur_value2 = self.criticize2(state, action)
        value_loss = self.criterion(cur_value, y_i.to(cur_value.device))
        value_loss2 = self.criterion(cur_value2, y_i.to(cur_value.device))

        if self.visualize:
            self.visualize_model(value_loss, "critic")

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()
            self.critic2.zero_grad()
            value_loss2.backward()
            self.critic2_optim.step()

        # Update actor network
        cur_action = self.action_transform_func(self.act(state), state, *others)
        act_value = self.criticize(state, cur_action)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        if self.visualize:
            self.visualize_model(act_policy_loss, "actor")

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            self.actor_optim.step()

        # Update target networks
        if update_target:
            soft_update(self.actor_target, self.actor, self.update_rate)
            soft_update(self.critic_target, self.critic, self.update_rate)
            soft_update(self.critic2_target, self.critic2, self.update_rate)

        # use .item() to prevent memory leakage
        return (-act_policy_loss.item(),
                (value_loss.item() + value_loss2.item()) / 2)

    @staticmethod
    def _policy_noise_function(actions, *_):
        return actions

    def load(self, model_dir: str, network_map: Dict[str, str] = None,
             version: int = -1):
        # DOC INHERITED
        TorchFramework.load(self, model_dir, network_map, version)
        with t.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic2, self.critic2_target)
