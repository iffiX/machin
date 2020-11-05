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


def _make_tensor_from_batch(batch: List[Any], device, concatenate):
    """
    Used to convert compact every attribute of every step of a whole episode
    into a single tensor.

    Args:
        batch: A list of tensor or scalar. If elements are tensors, they will
            be concatenated in dimension 0.
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
            return t.cat([it.to(device) for it in batch], dim=0).to(device)
        else:
            return t.tensor(batch, device=device).view(batch_size, -1)
    else:
        return batch


class EpisodeTransition(Transition):
    """
    A transition class which allows storing the whole episode as a
    single transition object, the batch dimension will be used to
    stack all transition steps.
    """
    def _check_validity(self):
        """
        Disable checking for batch size in the base :class:`.Transition`
        """
        super(Transition, self)._check_validity()


class EpisodeDistributedBuffer(DistributedBuffer):
    """
    A distributed buffer which stores each episode as a transition
    object inside the buffer.
    """
    def append(self, transition: Dict,
               required_attrs=("state", "action", "next_state",
                               "reward", "terminal", "action_log_prob")):
        """
        Append a transition.

        Args:
            self: (todo): write your description
            transition: (todo): write your description
            required_attrs: (todo): write your description
        """
        transition = EpisodeTransition(**transition)
        super(EpisodeDistributedBuffer, self)\
            .append(transition, required_attrs=required_attrs)


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
                 impala_group: RpcGroup,
                 model_server: Tuple[PushPullModelServer],
                 *_,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple] = (),
                 lr_scheduler_kwargs: Tuple[Dict, Dict] = (),
                 batch_size: int = 5,
                 learning_rate: float = 0.001,
                 isw_clip_c: float = 1.0,
                 isw_clip_rho: float = 1.0,
                 entropy_weight: float = None,
                 value_weight: float = 0.5,
                 gradient_max: float = np.inf,
                 discount: float = 0.99,
                 replay_size: int = 500,
                 visualize: bool = False,
                 **__):
        """
        Note:
            Please make sure isw_clip_rho >= isw_clip_c
        Args:
            actor: Actor network module.
            critic: Critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            impala_group: Group of all processes using the IMPALA framework,
                including all samplers and trainers.
            model_server: Custom model sync server accessor for ``actor``.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used during training.
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

        self.impala_group = impala_group

        self.actor = actor
        self.critic = critic
        self.actor_optim = optimizer(self.actor.parameters(),
                                     lr=learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(),
                                      lr=learning_rate)
        self.replay_buffer = EpisodeDistributedBuffer(
            buffer_name="buffer", group=impala_group,
            buffer_size=replay_size
        )
        self.is_syncing = True
        self.actor_model_server = model_server[0]

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

    def set_sync(self, is_syncing):
        """
        Sets the syncing.

        Args:
            self: (todo): write your description
            is_syncing: (todo): write your description
        """
        self.is_syncing = is_syncing

    def manual_sync(self):
        """
        Manages up the actor.

        Args:
            self: (todo): write your description
        """
        self.actor_model_server.pull(self.actor)

    def act(self, state: Dict[str, Any], *_, **__):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Anything produced by actor.
        """
        if self.is_syncing:
            self.actor_model_server.pull(self.actor)
        return safe_call(self.actor, state)

    def _eval_act(self,
                  state: Dict[str, Any],
                  action: Dict[str, Any],
                  *_, **__):
        """
        Use actor network to evaluate the log-likelihood of a given
        action in the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(self.actor, state, action)

    def _criticize(self, state: Dict[str, Any], *_, **__):
        """
        Use critic network to evaluate current value.

        Returns:
            Value of shape ``[batch_size, 1]``
        """
        return safe_call(self.critic, state)[0]

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

        # therefore, only one entry will be stored into the buffer
        # each entry in the buffer is of shape [episode_length, ...]

        # In other frameworks. each entry in the buffer is of shape
        # [1, ...]
        for k, v in episode[0].items():
            if k in ("state", "action", "next_state"):
                tmp_dict = {}
                for sub_k in v.keys():
                    tmp_dict[sub_k] = _make_tensor_from_batch(
                        [item[k][sub_k] for item in episode],
                        self.replay_buffer.buffer_device, True
                    )
                cc_episode[k] = tmp_dict
            elif k in ("reward", "terminal", "action_log_prob"):
                cc_episode[k] = _make_tensor_from_batch(
                    [item[k] for item in episode],
                    self.replay_buffer.buffer_device, True
                )
            else:
                # currently, additional attributes are not supported.
                pass

        self.replay_buffer.append(cc_episode, required_attrs=(
            "state", "action", "next_state", "reward",
            "action_log_prob", "terminal"
        ))

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

        # Note: each episode is stored as a single sample entry,
        # the second dimension of all attributes is the length of episode,
        # the first dimension is always 1.

        # `batch_size` here means the number of episodes sampled, not
        # the number of steps sampled.

        # `concatenate` is False, because the length of each episode
        # might be different.
        self.actor.train()
        self.critic.train()
        batch_size, (state, action, reward, next_state,
                     terminal, action_log_prob) = \
            self.replay_buffer.sample_batch(self.batch_size,
                                            concatenate=False,
                                            device="cpu",
                                            sample_attrs=[
                                                "state", "action", "reward",
                                                "next_state", "terminal",
                                                "action_log_prob"],
                                            additional_concat_attrs=[
                                                "action_log_prob"
                                            ])
        # `state`, `action` and `next_state` should be dicts like:
        # {"attr1": [Tensor(ep1_length, ...),
        #            Tensor(ep2_length, ...)]}

        # `terminal`, `reward`, `action_log_prob` should be lists like:
        # [Tensor(ep1_length, 1), (ep2_length, 1)]

        # chain steps of all episodes together, make them look like:
        # ep1_step1, ep1_step2, ..., ep1_stepN, ep2_step1, ep2_step2 ...

        # store the length of each episode, so that we can find boundaries
        # between two episodes inside the chained "sample"
        all_length = [tensor.shape[0] for tensor in terminal]
        sum_length = sum(all_length)

        for major_attr in (state, action, next_state):
            for k, v in major_attr.items():
                major_attr[k] = t.cat(v, dim=0)
                assert major_attr[k].shape[0] == sum_length

        terminal = t.cat(terminal, dim=0).view(sum_length, 1)
        reward = t.cat(reward, dim=0).view(sum_length, 1)
        action_log_prob = t.cat(action_log_prob, dim=0).view(sum_length, 1)

        # Below are the v-trace process

        # Calculate c and rho first, because there is no dependency
        # between vector elements.
        _, cur_action_log_prob, entropy, *__ = self._eval_act(state, action)
        cur_action_log_prob = cur_action_log_prob.view(sum_length, 1).to("cpu")
        entropy = entropy.view(sum_length, 1).to("cpu")

        # similarity = pi(a_t|x_t)/mu(a_t|x_t)
        sim = t.exp(cur_action_log_prob - action_log_prob)
        c = t.min(t.full(sim.shape, self.isw_clip_c), sim)
        rho = t.min(t.full(sim.shape, self.isw_clip_rho), sim)

        # calculate delta V
        # delta_t V = rho_t(r_t + gamma * V(x_{t+1}) - V(x_t))
        # boundary elements (i.e, ep1_stepN) will have V(x_{t+1}) = 0
        value = self._criticize(state).view(sum_length, 1).to("cpu")
        next_value = self._criticize(next_state).view(sum_length, 1).to("cpu")
        next_value[terminal] = 0
        delta_v = rho * (reward + self.discount * next_value - value)

        # calculate v_s

        # vss is v_s shifted left by 1 element, i.e. becomes v_{s+1}
        # boundary elements (i.e, ep1_stepN) will have v_{s+1} = 0

        # do reversed cumulative product for each episode segment
        with t.no_grad():
            vs = t.zeros(value.shape)
            vss = t.zeros(value.shape)
            offset = 0
            for ep_len in all_length:
                # the last v_s of each episode should be 0
                # or V_t - rho_t * (r_t - V_t)? (since Vt+1 = 0)
                # Implementations such as
                # https://github.com/junjungoal/IMPALA-pytorch/blob/master
                # /agents/learner.py   use the first case, 0
                # 0 works well when rho=c=1 or 1 > rho >= c
                for rev_step in reversed(range(ep_len - 1)):
                    idx = offset + rev_step
                    vs[idx] = (value[idx] + delta_v[idx] +
                               self.discount * c[idx] *
                               (vs[idx + 1] - value[idx + 1]))
                # shift v_s to get v_{s+1}
                vss[offset: offset + ep_len - 1] = \
                    vs[offset + 1: offset + ep_len]

                # update offset
                offset += ep_len

        act_policy_loss = -(rho.detach() * cur_action_log_prob *
                            (reward + self.discount * vss - value).detach())
        if self.entropy_weight is not None:
            act_policy_loss += self.entropy_weight * entropy
        act_policy_loss = act_policy_loss.sum()

        value_loss = self.criterion(value, vs.to(value.device))

        # Update actor network
        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_max
            )
            self.actor_optim.step()

        # Update critic network
        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.grad_max
            )
            self.critic_optim.step()

        # push actor model for samplers
        if isinstance(self.actor,
                      (nn.parallel.DataParallel,
                       nn.parallel.DistributedDataParallel)):
            self.actor_model_server.push(self.actor.module,
                                         pull_on_fail=False)
        else:
            self.actor_model_server.push(self.actor)

        self.actor.eval()
        self.critic.eval()
        return -act_policy_loss.item(), value_loss.item()

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()
