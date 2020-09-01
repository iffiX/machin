from machin.frame.buffers.prioritized_buffer import PrioritizedBuffer
from machin.utils.logging import default_logger
# pylint: disable=wildcard-import, unused-wildcard-import
from .ddpg import *


class DDPGPer(DDPG):
    """
    DDPG with prioritized experience replay.

    Warning:
        Your criterion must return a tensor of shape ``[batch_size,1]``
        when given two tensors of shape ``[batch_size,1]``, since we
        need to multiply the loss with importance sampling weight
        element-wise.

        If you are using loss modules given by pytorch. It is always
        safe to use them without any modification.
    """
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 actor_target: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 criterion,
                 *_,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple] = None,
                 lr_scheduler_kwargs: Tuple[Dict, Dict] = None,
                 batch_size: int = 100,
                 update_rate: float = 0.005,
                 actor_learning_rate: float = 0.0005,
                 critic_learning_rate: float = 0.001,
                 discount: float = 0.99,
                 gradient_max: float = np.inf,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 visualize: bool = False,
                 visualize_dir: str = "",
                 **__):
        # DOC INHERITED
        super(DDPGPer, self).__init__(
            actor, actor_target, critic, critic_target, optimizer, criterion,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            batch_size=batch_size,
            update_rate=update_rate,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            discount=discount,
            gradient_max=gradient_max,
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=(PrioritizedBuffer(replay_size, replay_device)
                           if replay_buffer is None
                           else replay_buffer),
            visualize=visualize,
            visualize_dir=visualize_dir
        )
        # reduction must be None
        if not hasattr(self.criterion, "reduction"):
            raise RuntimeError("Criterion does not have the "
                               "'reduction' property")
        else:
            if hasattr(self.criterion, "reduction"):
                # A loss defined in ``torch.nn.modules.loss``
                if self.criterion.reduction != "none":
                    default_logger.warning(
                        "The reduction property of criterion is not 'none', "
                        "automatically corrected."
                    )
                    self.criterion.reduction = "none"

    def update(self,
               update_value=True,
               update_policy=True,
               update_target=True,
               concatenate_samples=True,
               **__):
        # DOC INHERITED
        self.actor.train()
        self.critic.train()
        (batch_size,
         (state, action, reward, next_state, terminal, others),
         index, is_weight) = \
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
            next_action = self.action_transform_function(
                self._act(next_state, True), next_state, others
            )
            next_value = self._criticize(next_state, next_action, True)
            next_value = next_value.view(batch_size, -1)
            y_i = self.reward_function(
                reward, self.discount, next_value, terminal, others
            )

        # critic loss
        cur_value = self._criticize(state, action)
        value_loss = self.criterion(cur_value, y_i.to(cur_value.device))
        value_loss = (value_loss *
                      t.from_numpy(is_weight).view([batch_size, 1])
                      .to(value_loss.device))
        value_loss = value_loss.mean()

        if self.visualize:
            self.visualize_model(value_loss, "critic", self.visualize_dir)

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.grad_max
            )
            self.critic_optim.step()

        # actor loss
        cur_action = self.action_transform_function(
            self._act(state), state, others
        )
        act_value = self._criticize(state, cur_action)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        # update priority
        abs_error = (t.sum(t.abs(act_value - y_i.to(act_value.device)), dim=1)
                     .flatten().detach().cpu().numpy())
        self.replay_buffer.update_priority(abs_error, index)

        if self.visualize:
            self.visualize_model(act_policy_loss, "actor", self.visualize_dir)

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_max
            )
            self.actor_optim.step()

        # Update target networks
        if update_target:
            soft_update(self.actor_target, self.actor, self.update_rate)
            soft_update(self.critic_target, self.critic, self.update_rate)

        self.actor.eval()
        self.critic.eval()
        # use .item() to prevent memory leakage
        return -act_policy_loss.item(), value_loss.item()
