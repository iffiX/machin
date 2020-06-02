# pylint: disable=wildcard-import, unused-wildcard-import
from .ddpg import *


class HDDPG(DDPG):
    """
    HDDPG framework.
    """
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 actor_target: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 criterion: Callable,
                 *_,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple] = (),
                 lr_scheduler_kwargs: Tuple[Dict, Dict] = (),
                 batch_size: int = 100,
                 update_rate: float = 0.005,
                 learning_rate: float = 0.001,
                 discount: float = 0.99,
                 q_increase_rate: float = 1.0,
                 q_decrease_rate: float = 0.5,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 reward_func: Callable = None,
                 action_trans_func: Callable = None,
                 visualize: bool = False,
                 **__):
        """
        Args:
            q_increase_rate: The increase ratio multiplied to q value
                and target value difference when difference is positive.
            q_decrease_rate: The decrease ratio multiplied to q value
                and target value difference when difference is negative.

        See Also:
            :class:`machin.models.frames.algorithms.DDPG`
        """
        self.q_increase_rate = q_increase_rate
        self.q_decrease_rate = q_decrease_rate
        super(HDDPG, self).__init__(actor, actor_target, critic, critic_target,
                                    optimizer, criterion,
                                    lr_scheduler=lr_scheduler,
                                    lr_scheduler_args=lr_scheduler_args,
                                    lr_scheduler_kwargs=lr_scheduler_kwargs,
                                    batch_size=batch_size,
                                    update_rate=update_rate,
                                    learning_rate=learning_rate,
                                    discount=discount,
                                    replay_size=replay_size,
                                    replay_device=replay_device,
                                    replay_buffer=replay_buffer,
                                    reward_func=reward_func,
                                    action_trans_func=action_trans_func,
                                    visualize=visualize)

    def update(self,
               update_value=True,
               update_policy=True,
               update_targets=True,
               concatenate_samples=True,
               **__):
        """
        Update network weights by sampling from replay buffer.

        Args:
            update_value: Whether update the Q network.
            update_policy: Whether update the actor network.
            update_targets: Whether update targets.
            concatenate_samples: Whether concatenate the samples.

        Returns:
            (mean value of estimated policy value, value loss)
        """
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
            next_action = self.action_transform_func(self.act(next_state, True),
                                                     next_state,
                                                     *others)
            next_value = self.criticize(next_state, next_action, True)
            next_value = next_value.view(batch_size, -1)
            y_i = self.reward_func(reward, self.discount, next_value,
                                   terminal, *others)

        cur_value = self.criticize(state, action)
        value_diff = y_i.to(cur_value.device) - cur_value
        value_change = t.where(value_diff > 0,
                               value_diff * self.q_increase_rate,
                               value_diff * self.q_decrease_rate)
        value_loss = self.criterion(cur_value,
                                    (cur_value + value_change).detach())

        if self.visualize:
            self.visualize_model(value_loss, "critic")

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

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
        if update_targets:
            soft_update(self.actor_target, self.actor, self.update_rate)
            soft_update(self.critic_target, self.critic, self.update_rate)

        # use .item() to prevent memory leakage
        return -act_policy_loss.item(), value_loss.item()
