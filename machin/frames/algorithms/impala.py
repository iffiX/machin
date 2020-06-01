import numpy as np
import torch as t
import torch.nn as nn

from models.frames.buffers.buffer_d import Transition, DistributedBuffer
from models.nets.base import NeuralNetworkModule
from typing import Union, Dict, List
from .base import TorchFramework
from .utils import safe_call

from utils.visualize import visualize_graph
from utils.parallel.server import SimplePushPullServer


def slice(dict, index, dim=1):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k] = v.select(dim, index)
    return new_dict


class IMPALA_Buffer(DistributedBuffer):
    @staticmethod
    def make_tensor_from_batch(batch, device, concatenate):
        if len(batch) == 0:
            return None
        if concatenate:
            item = batch[0]
            batch_size = len(batch)
            if t.is_tensor(item):
                batch = [it.to(device) for it in batch]
                if item.dim() >= 3:
                    # ignore size difference in dimension 1 and pad with 0
                    # if there if no size difference, result should be the same as cat
                    pad_length = max([it.shape[1] for it in batch])
                    remain_shape = [item.shape[2:]]
                    result = t.zeros([batch_size, pad_length] + remain_shape,
                                     dtype=item.dtype, device=device)
                    for it, idx in zip(batch, range(batch_size)):
                        result[idx, :it.shape[1]] = it
                    return result
                else:
                    return t.cat(batch, dim=0).to(device)
            else:
                return t.tensor(batch, device=device).view(batch_size, -1)
        else:
            return batch


class IMPALA(TorchFramework):
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 worker_group,
                 trainer_group,
                 pull_function=None,
                 push_function=None,
                 isw_clip_c=0.8,
                 isw_clip_rho=0.9,
                 entropy_weight=None,
                 value_weight=0.5,
                 gradient_max=np.inf,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=5,
                 discount=0.99,
                 replay_size=50,
                 *_, **__):
        """
        Initialize IMPALA framework.
        """
        self.batch_size = batch_size
        self.discount = discount
        self.rpb = IMPALA_Buffer(buffer_size=replay_size,
                                 buffer_group=worker_group)

        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.grad_max = gradient_max
        self.isw_clip_c = isw_clip_c
        self.isw_clip_rho = isw_clip_rho

        self.actor = actor
        self.critic = critic
        self.actor_optim = optimizer(self.actor.parameters(), learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), learning_rate)

        if push_function is None or pull_function is None:
            self.pp = SimplePushPullServer(trainer_group)
            self.pull_function = self.pp.pull
            self.push_function = self.pp.push
        else:
            self.pull_function = pull_function
            self.push_function = push_function

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(self.actor_optim, *lr_scheduler_params[0])
            self.critic_lr_sch = lr_scheduler(self.critic_optim, *lr_scheduler_params[1])

        self.criterion = criterion

        super(IMPALA, self).__init__()
        self.set_top(["actor", "critic"])
        self.set_restorable(["actor", "critic"])

    def act(self, state):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Anything produced by actor.
        """
        self.pull_function(self.actor, "actor")
        return safe_call(self.actor, state)

    def eval_act(self, state, action):
        """
        Use actor network to evaluate the log-likelihood of a given action in the current state.

        Returns:
            Anything produced by actor.
        """
        self.pull_function(self.actor, "actor")
        return safe_call(self.actor, state, action)

    def criticize(self, state):
        """
        Use critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        """
        self.pull_function(self.critic, "critic")
        return safe_call(self.critic, state)

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        if not isinstance(episode[0], Transition):
            episode = [Transition(**trans) for trans in episode]

        cc_episode = {}
        for k, v in episode[0].items():
            if k in ("state", "action", "next_state"):
                tmp_dict = {}
                for sub_k in v.keys():
                    tmp_dict[sub_k] = self.rpb.make_tensor_from_batch(
                        [item[k][sub_k] for item in episode],
                        self.rpb.buffer_device, True
                    ).unsqueeze(0)
                cc_episode[k] = tmp_dict
            elif k in ("reward", "terminal"):
                cc_episode[k] = self.rpb.make_tensor_from_batch(
                    [item[k] for item in episode],
                    self.rpb.buffer_device, True
                ).unsqueeze(0)
            else:
                cc_episode[k] = self.rpb.make_tensor_from_batch(
                    [item[k] for item in episode],
                    self.rpb.buffer_device, False
                )

        self.rpb.append(cc_episode)

    def get_replay_buffer(self):
        return self.rpb

    def update(self, update_value=True, update_policy=True):
        # sample a batch
        batch_size, (state, action, reward, next_state, terminal, action_log_prob) = \
            self.rpb.sample_batch(self.batch_size,
                                  concatenate=True,
                                  device="cpu",
                                  sample_attrs=["state", "action", "reward", "next_state",
                                               "terminal", "action_log_prob"],
                                  additional_concat_attrs=["action_log_prob"])
        step_len = reward.shape[1]
        term = t.zeros([batch_size], dtype=t.bool)
        values = []
        log_probs = []
        entropies = []
        c = []
        rho = []
        clip_c = t.full(action_log_prob.shape, self.isw_clip_c)
        clip_rho = t.full(action_log_prob.shape, self.isw_clip_rho)

        # pre process
        for step in range(step_len):
            term = term | terminal[:, step]
            terminal[:, step] = term
            value = self.criticize(slice(state, step)).to("cpu")

            if self.entropy_weight is not None:
                _, new_action_log_prob, new_action_entropy, *_ = \
                    self.eval_act(slice(state, step), slice(action, step))
                entropies.append(new_action_entropy)
            else:
                _, new_action_log_prob, *_ = \
                    self.eval_act(slice(state, step), slice(action, step))

            new_action_log_prob = new_action_log_prob.to("cpu")
            # importance sampling weight
            is_weight = t.exp(new_action_log_prob - action_log_prob)

            values.append(value)
            log_probs.append(new_action_log_prob)
            c.append(t.min(clip_c, is_weight))
            rho.append(t.min(clip_rho, is_weight))

        act_policy_loss = t.zeros([1])

        # v-trace, ``vs`` is v-trace target
        vs = t.zeros([step_len, batch_size, 1], device="cpu")
        for rev_step in reversed(range(step_len - 1)):
            # terminal is not in the original essay? Maybe it is useful?
            delta_v = rho[rev_step] * (reward[:, rev_step] * (1 - terminal[:, rev_step])
                                       + self.discount * values[rev_step + 1]
                                       - values[rev_step])
            advantage = rho[rev_step] * (reward[:, rev_step] * (1 - terminal[:, rev_step])
                                         + self.discount * vs[rev_step + 1]
                                         - values[rev_step])
            vs[rev_step] = \
                values[rev_step] + delta_v + \
                self.discount * c[rev_step] * (vs[rev_step + 1] - values[rev_step + 1])

            act_policy_loss += log_probs[rev_step] * advantage.detach()

        act_policy_loss += self.entropy_weight * t.sum(t.stack(entropies))
        act_policy_loss = act_policy_loss.sum()

        value_loss = self.criterion(t.stack(values), vs.transpose(0, 1).detach())

        # Update actor network
        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_max)
            self.actor_optim.step()
            self.push_function(self.actor, "actor")

        # Update critic network
        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_max)
            self.critic_optim.step()
            self.push_function(self.critic, "critic")

        self.rpb.clear()
        return -act_policy_loss.item(), value_loss.item()

    def update_lr_scheduler(self):
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()
