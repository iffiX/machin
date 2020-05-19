from .dqn import *
from models.frames.buffers.prioritized_buffer import PrioritizedBuffer
from utils.logging import default_logger
import torch.nn as nn


class RAINBOW(DQN):
    def __init__(self,
                 qnet: Union[NeuralNetworkModule, nn.Module],
                 qnet_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 v_min,
                 v_max,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=100,
                 update_rate=1.0,
                 discount=0.99,
                 replay_size=500000,
                 replay_device="cpu",
                 reward_func=None):
        """
        RAINBOW DQN implementation.
        """
        super(RAINBOW, self).__init__(
            qnet, qnet_target, optimizer, None,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            batch_size=batch_size,
            update_rate=update_rate,
            discount=discount,
            replay_size=replay_size,
            replay_device=replay_device,
            reward_func=reward_func
        )
        self.v_min = v_min
        self.v_max = v_max
        self.rpb = PrioritizedBuffer(replay_size, replay_device)

    def update(self, update_value=True, update_targets=True, concatenate_samples=True):
        """
        Update network weights by sampling from replay buffer.

        Returns:
            (mean value of estimated policy value, value loss)
        """
        batch_size, (state, action, reward, next_state, terminal, *others), index, is_weight = \
            self.rpb.sample_batch(self.batch_size, concatenate_samples,
                                  sample_keys=["state", "action", "reward", "next_state", "terminal", "*"])

        # q_dist is the distribution of q values, with shape (batch_size, action_num, atom_num)
        q_dist = self.criticize(state)
        atom_num = q_dist.shape(-1)
        action["action"] = action["action"].unsqueeze(-1).expand(batch_size, -1, atom_num)

        # shape: (batch_size, atom_num)
        action_value_dist = q_dist.gather(dim=1, index=action["action"]).squeeze(1)

        # support vector, shape: (atom_num)
        q_dist_support = t.linspace(self.v_min, self.v_max, atom_num)

        with torch.no_grad():
            next_q_dist = self.criticize(next_state)
            target_next_q_dist = self.criticize(next_state, True)
            next_q_value = (next_q_dist * q_dist_support).sum(dim=2)
            next_action = t.max(next_q_value, dim=1).indicies.unsqueeze(1)

            # shape: (batch_size, atom_num)
            target_next_q_dist = target_next_q_dist.gather(dim=1, index=next_action).squeeze(1)

        # shape: (batch_size, atom_num)
        q_dist_support = q_dist_support.unsqueeze(dim=0).expand_as(target_next_q_dist)

        # T_z is the bellman update for atom z_j
        # shape: (batch_size, atom_num)
        T_z = self.reward_func(reward.view, self.discount, q_dist_support,
                               terminal, *others)
        T_z = T_z.clamp(self.v_min, self.v_max)

        # delta_z is the interval length of each atom
        delta_z = (self.v_max - self.v_min) / (atom_num - 1.0)

        # b is the distance of T_z to v_min, l and u are upper and lower atom indexes
        # b, l, u shape: (batch_size, atom_num)
        b = (T_z - self.v_min) / delta_z
        l, u = b.floor(), b.ceil()
        l_idx, l_dist = l.long().view(-1).cpu(), b - l
        u_idx, u_dist = u.long().view(-1).cpu(), u - b
        l_weight = (l_dist * target_next_q_dist).view(-1).cpu()
        u_weight = (u_dist * target_next_q_dist).view(-1).cpu()

        offset = t.arange(0, batch_size * atom_num, atom_num).view([-1, 1]).expand(batch_size, atom_num).view(-1)
        # distribute T_z probability to its nearest upper and lower atom neighbors,
        # using its distance to them.
        # shape: (batch_size, atom_num)
        # Note: index_add_ on cuda is nondeterministic. will introduce noise
        target_dist = t.zeros_like(target_next_q_dist)
        target_dist.index_add_(dim=0, index=l_idx.view(-1) + offset, source=l_weight)
        target_dist.index_add_(dim=0, index=u_idx.view(-1) + offset, source=u_weight)

        # target_dist is equivalent to y_i in original dqn
        value_loss = -(target_dist * q_dist.log().cpu())
        value_loss = value_loss.sum(dim=1)

        abs_error = (t.abs(value_loss) + 1e-6).flatten().numpy()
        self.rpb.update_priority(abs_error, index)

        value_loss = (value_loss * t.from_numpy(is_weight).view([batch_size, 1])).mean()

        if update_value:
            self.qnet.zero_grad()
            value_loss.backward()
            self.qnet_optim.step()

        # Update target networks
        if update_targets:
            soft_update(self.qnet_target, self.qnet, self.update_rate)

        # use .item() to prevent memory leakage
        return value_loss.item()

