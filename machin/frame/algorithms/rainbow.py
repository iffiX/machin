from machin.frame.buffers.prioritized_buffer import PrioritizedBuffer
# pylint: disable=wildcard-import, unused-wildcard-import
from .dqn import *


class RAINBOW(DQN):
    """
    RAINBOW DQN framework.
    """
    def __init__(self,
                 qnet: Union[NeuralNetworkModule, nn.Module],
                 qnet_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 value_min,
                 value_max,
                 *_,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple] = (),
                 lr_scheduler_kwargs: Tuple[Dict, Dict] = (),
                 batch_size: int = 100,
                 update_rate: float = 0.005,
                 learning_rate: float = 0.001,
                 discount: float = 0.99,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 reward_func: Callable = None,
                 visualize: bool = False,
                 **__):
        """
        RAINBOW framework is described in
        `this <https://arxiv.org/abs/1710.02298>`__ essay.

        See Also:
            :class:`.DQN`

        Args:
            qnet: Q network module.
            qnet_target: Target Q network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            value_min: Minimum of value domain.
            value_max: Maximum of value domain.
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
            mode: one of ``"vanilla", "fixed_target", "double"``.
            visualize: Whether visualize the network flow in the first pass.
        """
        super(RAINBOW, self).__init__(
            qnet, qnet_target, optimizer, criterion,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            batch_size=batch_size,
            update_rate=update_rate,
            discount=discount,
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=(PrioritizedBuffer(replay_size, replay_device)
                           if replay_buffer is None
                           else replay_buffer),
            reward_func=reward_func,
            visualize=visualize
        )
        self.v_min = value_min
        self.v_max = value_max

    def update(self,
               update_value=True,
               update_target=True,
               concatenate_samples=True,
               **__):
        # DOC INHERITED
        # pylint: disable=invalid-name
        (batch_size,
         (state, action, reward, next_state, terminal, *others),
         index, is_weight) = \
            self.replay_buffer.sample_batch(self.batch_size,
                                            concatenate_samples,
                                            sample_attrs=[
                                                "state", "action",
                                                "reward", "next_state",
                                                "terminal", "*"
                                            ])

        # q_dist is the distribution of q values,
        # with shape [batch_size, action_num, atom_num]
        q_dist = self.criticize(state)
        atom_num = q_dist.shape(-1)
        action["action"] = (action["action"]
                            .unsqueeze(-1)
                            .expand(batch_size, -1, atom_num))

        # support vector, shape: [atom_num]
        q_dist_support = t.linspace(self.v_min, self.v_max, atom_num)

        with t.no_grad():
            next_q_dist = self.criticize(next_state)
            target_next_q_dist = self.criticize(next_state, True)
            next_q_value = (next_q_dist * q_dist_support).sum(dim=2)
            next_action = t.max(next_q_value, dim=1)[1].unsqueeze(1)

            # shape: [batch_size, atom_num]
            target_next_q_dist = (target_next_q_dist
                                  .gather(dim=1, index=next_action)
                                  .squeeze(1))

        # shape: [batch_size, atom_num]
        q_dist_support = (q_dist_support
                          .unsqueeze(dim=0)
                          .expand_as(target_next_q_dist))

        # T_z is the bellman update for atom z_j
        # shape: [batch_size, atom_num]
        T_z = self.reward_func(reward.view, self.discount, q_dist_support,
                               terminal, *others)
        T_z = T_z.clamp(self.v_min, self.v_max)

        # delta_z is the interval length of each atom
        delta_z = (self.v_max - self.v_min) / (atom_num - 1.0)

        # b is the distance of T_z to v_min,
        # l and u are upper and lower atom indexes
        # b, l, u shape: [batch_size, atom_num]
        b = (T_z - self.v_min) / delta_z
        l, u = b.floor(), b.ceil()
        l_idx, l_dist = l.long().view(-1).cpu(), b - l
        u_idx, u_dist = u.long().view(-1).cpu(), u - b
        l_weight = (l_dist * target_next_q_dist).view(-1).cpu()
        u_weight = (u_dist * target_next_q_dist).view(-1).cpu()

        offset = (t.arange(0, batch_size * atom_num, atom_num)
                  .view([-1, 1])
                  .expand(batch_size, atom_num)
                  .view(-1))

        # distribute T_z probability to its nearest upper
        # and lower atom neighbors, using its distance to them.
        # shape: [batch_size, atom_num]
        # Note: index_add_ on CUDA is non-deterministic. will introduce noise
        target_dist = t.zeros_like(target_next_q_dist)
        target_dist.index_add_(dim=0, index=l_idx.view(-1) + offset,
                               source=l_weight)
        target_dist.index_add_(dim=0, index=u_idx.view(-1) + offset,
                               source=u_weight)

        # target_dist is equivalent to y_i in original dqn
        value_loss = -(target_dist * q_dist.log().cpu())
        value_loss = value_loss.sum(dim=1)

        abs_error = (t.abs(value_loss) + 1e-6).flatten().numpy()
        self.replay_buffer.update_priority(abs_error, index)

        value_loss = (value_loss *
                      t.from_numpy(is_weight).view([batch_size, 1])).mean()

        if self.visualize:
            self.visualize_model(value_loss, "qnet")

        if update_value:
            self.qnet.zero_grad()
            value_loss.backward()
            self.qnet_optim.step()

        # Update target Q network
        if update_target:
            soft_update(self.qnet_target, self.qnet, self.update_rate)

        # use .item() to prevent memory leakage
        return value_loss.item()
