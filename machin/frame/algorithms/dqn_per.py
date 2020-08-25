from machin.frame.buffers.prioritized_buffer import PrioritizedBuffer
from machin.utils.logging import default_logger
# pylint: disable=wildcard-import, unused-wildcard-import
from .dqn import *


class DQNPer(DQN):
    """
    DQN with prioritized replay. It is based on Double DQN.

    Warning:
        Your criterion must return a tensor of shape ``[batch_size,1]``
        when given two tensors of shape ``[batch_size,1]``, since we
        need to multiply the loss with importance sampling weight
        element-wise.

        If you are using loss modules given by pytorch. It is always
        safe to use them without any modification.
    """

    def __init__(self,
                 qnet: Union[NeuralNetworkModule, nn.Module],
                 qnet_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 criterion: Callable,
                 *_,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple] = None,
                 lr_scheduler_kwargs: Tuple[Dict] = None,
                 batch_size: int = 100,
                 update_rate: float = 0.005,
                 learning_rate: float = 0.001,
                 discount: float = 0.99,
                 gradient_max: float = np.inf,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 visualize: bool = False,
                 visualize_dir: str = "",
                 **__):
        # DOC INHERITED
        super(DQNPer, self).__init__(
            qnet, qnet_target, optimizer, criterion,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            batch_size=batch_size,
            update_rate=update_rate,
            learning_rate=learning_rate,
            discount=discount,
            gradient_max=gradient_max,
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=(PrioritizedBuffer(replay_size, replay_device)
                           if replay_buffer is None
                           else replay_buffer),
            mode="double",
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
               update_target=True,
               concatenate_samples=True,
               **__):
        # DOC INHERITED
        self.qnet.train()
        (batch_size,
         (state, action, reward, next_state, terminal, others),
         index, is_weight) = \
            self.replay_buffer.sample_batch(self.batch_size,
                                            concatenate_samples,
                                            sample_method="random_unique",
                                            sample_attrs=[
                                                "state", "action",
                                                "reward", "next_state",
                                                "terminal", "*"
                                            ])

        with t.no_grad():
            next_q_value = self._criticize(next_state)
            target_next_q_value = self._criticize(next_state, True)
            target_next_q_value = target_next_q_value.gather(
                dim=1, index=t.max(next_q_value, dim=1)[1].unsqueeze(1))

        q_value = self._criticize(state)

        # gather requires long tensor, int32 is not accepted
        action_value = q_value.gather(dim=1,
                                      index=self.action_get_function(action)
                                      .to(device=q_value.device,
                                          dtype=t.long))

        # Generate value reference :math: `y_i`.
        y_i = self.reward_function(
            reward, self.discount, target_next_q_value, terminal, others
        )
        value_loss = self.criterion(action_value, y_i.to(action_value.device))
        value_loss = (value_loss *
                      t.from_numpy(is_weight).view([batch_size, 1])
                      .to(value_loss.device))
        value_loss = value_loss.mean()

        abs_error = (t.sum(t.abs(action_value - y_i.to(action_value.device)),
                           dim=1)
                     .flatten().detach().cpu().numpy())
        self.replay_buffer.update_priority(abs_error, index)

        if self.visualize:
            self.visualize_model(value_loss, "qnet", self.visualize_dir)

        if update_value:
            self.qnet.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.qnet.parameters(), self.grad_max
            )
            self.qnet_optim.step()

        # Update target Q network
        if update_target:
            soft_update(self.qnet_target, self.qnet, self.update_rate)

        self.qnet.eval()
        # use .item() to prevent memory leakage
        return value_loss.item()
