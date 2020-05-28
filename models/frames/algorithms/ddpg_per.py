from .ddpg import *
from models.frames.buffers.prioritized_buffer import PrioritizedBuffer
from utils.logging import default_logger
import torch.nn as nn


class DDPG_PER(DDPG):
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 actor_target: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=100,
                 update_rate=0.005,
                 discount=0.99,
                 replay_size=500000,
                 replay_device="cpu",
                 reward_func=None,
                 action_trans_func=None):
        """
        Initialize DDPG framework.
        """
        super(DDPG_PER, self).__init__(
            actor, actor_target, critic, critic_target, optimizer, criterion,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            batch_size=batch_size,
            update_rate=update_rate,
            discount=discount,
            replay_size=replay_size,
            replay_device=replay_device,
            reward_func=reward_func,
            action_trans_func=action_trans_func
        )
        self.rpb = PrioritizedBuffer(replay_size, replay_device)
        # reduction must be None
        if not hasattr(self.criterion, "reduction"):
            raise RuntimeError("Criterion must have the 'reduction' property")
        else:
            if self.criterion.reduction != "none":
                default_logger.warn("The reduction property of criterion is not 'none', "
                                    "automatically corrected.")
                self.criterion.reduction = "none"

    def update(self, update_value=True, update_policy=True, update_target=True, concatenate_samples=True):
        """
        Update network weights by sampling from replay buffer.

        Returns:
            (mean value of estimated policy value, value loss)
        """
        batch_size, (state, action, reward, next_state, terminal, *others), index, is_weight = \
            self.rpb.sample_batch(self.batch_size, concatenate_samples,
                                  sample_attrs=["state", "action", "reward", "next_state", "terminal", "*"])

        # Update critic network first
        # Generate value reference :math: `y_i` using target actor and target critic
        with torch.no_grad():
            next_action = self.action_trans_func(self.act(next_state, True), next_state, *others)
            next_value = self.criticize(next_state, next_action, True)
            next_value = next_value.view(batch_size, -1)
            y_i = self.reward_func(reward, self.discount, next_value, terminal, *others)

        # critic loss
        cur_value = self.criticize(state, action)
        value_loss = self.criterion(cur_value, y_i.to(cur_value.device)) * \
                     t.from_numpy(is_weight).view([batch_size, 1])
        value_loss = value_loss.mean()

        # actor loss
        cur_action = self.action_trans_func(self.act(state), state, *others)
        act_value = self.criticize(state, cur_action)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        # update priority
        abs_error = t.sum(t.abs(act_value - y_i), dim=1).flatten().cpu().numpy()
        self.rpb.update_priority(abs_error, index)

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            self.actor_optim.step()

        # Update target networks
        if update_target:
            soft_update(self.actor_target, self.actor, self.update_rate)
            soft_update(self.critic_target, self.critic, self.update_rate)

        # use .item() to prevent memory leakage
        return -act_policy_loss.item(), value_loss.item()