from typing import Union, Dict, List, Tuple, Callable, Any
import numpy as np
import torch as t
import torch.nn as nn

from machin.frame.buffers.buffer_d import Transition, DistributedBuffer
from machin.model.nets.base import NeuralNetworkModule
from .base import TorchFramework
from .utils import safe_call

from machin.parallel.server import PushPullModelServer
from machin.parallel.distributed import RpcGroup


def _slice(attr_dict, index, dim=1):
    # Used to split a specific slice from a transition main attribute
    new_dict = {}
    for k, v in attr_dict.items():
        new_dict[k] = v.select(dim, index)
    return new_dict


def _make_tensor_from_batch(batch: List[Any], device, concatenate):
    """
    Used to convert a whole episode to a single "Transition".

    Args:
        batch:
        device: Device to move tensors to.
        concatenate: Whether to perform concatenation or not, only True
            for major and sub attributes in ``Transition``.

    Returns:
        A tensor if ``concatenate`` is ``True``, otherwise the original List.
    """
    if len(batch) == 0:
        return None
    if concatenate:
        item = batch[0]
        batch_size = len(batch)
        if t.is_tensor(item):
            batch = [it.to(device) for it in batch]
            if item.dim() >= 3:
                # ignore size difference in dimension 1 and pad with 0
                # if there if no size difference, result should be the
                # same as cat
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
    """
    Massively parallel IMPALA framework.
    """

    _is_top = ["actor", "critic"]
    _is_restorable = ["actor", "critic"]

    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 criterion: Callable,
                 worker_group: RpcGroup,
                 trainer_group: RpcGroup,
                 *_,
                 pull_function: Callable = None,
                 push_function: Callable = None,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple] = (),
                 lr_scheduler_kwargs: Tuple[Dict, Dict] = (),
                 learning_rate=0.001,
                 isw_clip_c: float = 0.8,
                 isw_clip_rho: float = 0.9,
                 entropy_weight: float = None,
                 value_weight: float = 0.5,
                 gradient_max: float = np.inf,
                 discount: float = 0.99,
                 batch_size: int = 5,
                 replay_size: int = 500,
                 visualize: bool = False,
                 **__):
        """
        TODO: test IMPAlA and add more explanations in document.
        TODO: add visualization in update.

        Args:
            actor: Actor network module.
            critic: Critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            worker_group: Rpc group of roll out workers.
            trainer_group: Rpc group of model trainers.
            pull_function: User defined function used to pull the newest model.
            push_function: User defined function used to push the newest model.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            isw_clip_c: :math:`c` used in importance weight clipping.
            isw_clip_rho:
            entropy_weight: Weight of entropy in your loss function, a positive
                entropy weight will minimize entropy, while a negative one will
                maximize entropy.
            value_weight: Weight of critic value loss.
            gradient_max: Maximum gradient.
            discount: :math:`\\gamma` used in the bellman function.
            batch_size: Batch size used during training.
            replay_size: Size of the local replay buffer.
            visualize: Whether visualize the network flow in the first pass.
        """
        self.batch_size = batch_size
        self.discount = discount
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.grad_max = gradient_max
        self.isw_clip_c = isw_clip_c
        self.isw_clip_rho = isw_clip_rho
        self.visualize = visualize

        self.worker_group = worker_group
        self.trainer_group = trainer_group

        self.actor = actor
        self.critic = critic
        self.actor_optim = optimizer(self.actor.parameters(),
                                     lr=learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(),
                                      lr=learning_rate)
        self.replay_buffer = DistributedBuffer(buffer_size=replay_size,
                                               buffer_group=worker_group)

        if push_function is None or pull_function is None:
            self.pp = PushPullModelServer(trainer_group)
            self.pull_function = self.pp.pull
            self.push_function = self.pp.push
        else:
            self.pull_function = pull_function
            self.push_function = push_function

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(
                self.actor_optim,
                *lr_scheduler_args[0],
                **lr_scheduler_kwargs[0],
            )
            self.critic_lr_sch = lr_scheduler(
                self.critic_optim,
                *lr_scheduler_args[1],
                **lr_scheduler_kwargs[1]
            )

        self.criterion = criterion

        super(IMPALA, self).__init__()

    def act(self, state: Dict[str, Any], *_, **__):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(self.actor, state)

    def eval_act(self,
                 state: Dict[str, Any],
                 action: Dict[str, Any],
                 *_, **__):
        """
        Use actor network to evaluate the log-likelihood of a given
        action in the current state.

        Returns:
            Anything produced by actor.
        """
        self.pull_function(self.actor, "actor")
        return safe_call(self.actor, state, action)

    def criticize(self, state: Dict[str, Any], *_, **__):
        """
        Use critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        """
        self.pull_function(self.critic, "critic")
        return safe_call(self.critic, state)

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Warning:
            Not supported in IMPALA due to v-trace requirements.
        """
        raise NotImplementedError

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.
        """
        if not isinstance(episode[0], Transition):
            episode = [Transition(**trans) for trans in episode]

        cc_episode = {}
        # In order to compute v-trace, we must reshape the whole
        # episode to make it look like a single Transition, because
        # v-trace need to see all future rewards.
        for k, v in episode[0].items():
            if k in ("state", "action", "next_state"):
                tmp_dict = {}
                for sub_k in v.keys():
                    tmp_dict[sub_k] = _make_tensor_from_batch(
                        [item[k][sub_k] for item in episode],
                        self.replay_buffer.buffer_device, True
                    ).unsqueeze(0)
                cc_episode[k] = tmp_dict
            elif k in ("reward", "terminal"):
                cc_episode[k] = _make_tensor_from_batch(
                    [item[k] for item in episode],
                    self.replay_buffer.buffer_device, True
                ).unsqueeze(0)
            else:
                cc_episode[k] = _make_tensor_from_batch(
                    [item[k] for item in episode],
                    self.replay_buffer.buffer_device, False
                )

        self.replay_buffer.append(cc_episode)

    def update(self,
               update_value=True,
               update_policy=True,
               **__):
        """
        Update network weights by sampling from replay buffer.

        Note:
            Will always concatenate samples.

        Args:
            update_value: Whether to update the Q network.
            update_policy: Whether to update the actor network.

        Returns:
            mean value of estimated policy value, value loss
        """
        # sample a batch
        batch_size, (state, action, reward, next_state,
                     terminal, action_log_prob) = \
            self.replay_buffer.sample_batch(self.batch_size,
                                            concatenate=True,
                                            device="cpu",
                                            sample_attrs=[
                                                "state", "action", "reward",
                                                "next_state", "terminal",
                                                "action_log_prob"],
                                            additional_concat_attrs=[
                                                "action_log_prob"
                                            ])
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
            value = self.criticize(_slice(state, step)).to("cpu")

            if self.entropy_weight is not None:
                _, new_action_log_prob, new_action_entropy, *_ = \
                    self.eval_act(_slice(state, step), _slice(action, step))
                entropies.append(new_action_entropy)
            else:
                _, new_action_log_prob, *_ = \
                    self.eval_act(_slice(state, step), _slice(action, step))

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
            delta_v = rho[rev_step] * (reward[:, rev_step]
                                       * (1 - terminal[:, rev_step]) +
                                       self.discount * values[rev_step + 1] -
                                       values[rev_step])
            advantage = rho[rev_step] * (reward[:, rev_step]
                                         * (1 - terminal[:, rev_step]) +
                                         self.discount * vs[rev_step + 1] -
                                         values[rev_step])
            vs[rev_step] = (values[rev_step] +
                            delta_v +
                            self.discount
                            * c[rev_step]
                            * (vs[rev_step + 1] - values[rev_step + 1]))

            act_policy_loss += log_probs[rev_step] * advantage.detach()

        act_policy_loss += self.entropy_weight * t.sum(t.stack(entropies))
        act_policy_loss = act_policy_loss.sum()

        value_loss = self.criterion(t.stack(values).to(vs.device),
                                    vs.transpose(0, 1).detach())

        # Update actor network
        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_max
            )
            self.actor_optim.step()
            self.push_function(self.actor, "actor")

        # Update critic network
        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.grad_max
            )
            self.critic_optim.step()
            self.push_function(self.critic, "critic")

        return -act_policy_loss.item(), value_loss.item()

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()
