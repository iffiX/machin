from .dqn import *
from models.frames.buffers.prioritized_buffer import PrioritizedBuffer
from utils.logging import default_logger
import torch.nn as nn


class DQN_PER(DQN):
    def __init__(self,
                 qnet: Union[NeuralNetworkModule, nn.Module],
                 qnet_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
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
        Prioritized double DQN implementation.
        """
        super(DQN_PER, self).__init__(
            qnet, qnet_target, optimizer, criterion,
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
        self.rpb = PrioritizedBuffer(replay_size, replay_device)
        # reduction must be None
        if not hasattr(self.criterion, "reduction"):
            raise RuntimeError("Criterion must have the 'reduction' property")
        else:
            if self.criterion.reduction != "none":
                default_logger.warn("The reduction property of criterion is not 'none', "
                                    "automatically corrected.")
                self.criterion.reduction = "none"

    def update(self, update_value=True, update_target=True, concatenate_samples=True):
        """
        Update network weights by sampling from replay buffer.

        Returns:
            (mean value of estimated policy value, value loss)
        """
        batch_size, (state, action, reward, next_state, terminal, *others), index, is_weight = \
            self.rpb.sample_batch(self.batch_size, concatenate_samples,
                                  sample_keys=["state", "action", "reward", "next_state", "terminal", "*"])

        with torch.no_grad():
            next_q_value = self.criticize(next_state)
            target_next_q_value = self.criticize(next_state, True)
            target_next_q_value = target_next_q_value\
                .gather(dim=1, index=t.max(next_q_value, dim=1).indicies.unsqueeze(1))

        # Generate value reference :math: `y_i` using target actor and target qnet
        q_value = self.criticize(state)
        action_value = q_value.gather(dim=1, index=action["action"])

        y_i = self.reward_func(reward, self.discount, target_next_q_value, terminal, *others)\
                  .to(action_value.device)
        value_loss = self.criterion(action_value, y_i) * t.from_numpy(is_weight).view([batch_size, 1])
        value_loss = value_loss.mean()

        abs_error = t.sum(t.abs(action_value - y_i), dim=1).flatten().cpu().numpy()
        self.rpb.update_priority(abs_error, index)

        if update_value:
            self.qnet.zero_grad()
            value_loss.backward()
            self.qnet_optim.step()

        # Update target networks
        if update_target:
            soft_update(self.qnet_target, self.qnet, self.update_rate)

        # use .item() to prevent memory leakage
        return value_loss.item()
