# Adapted from https://github.com/modestyachts/ARS

from typing import Union, Dict, List, Tuple, Callable, Any
import copy
import torch as t
import torch.nn as nn
import numpy as np

from machin.model.nets.base import NeuralNetworkModule
from machin.parallel.server import PushPullModelServer
from machin.parallel.distributed import RpcGroup
from machin.utils.logging import default_logger
from .base import TorchFramework
from .utils import safe_call, safe_return


class RunningStat(object):
    """
    Running status estimator method by B. P. Welford
    described in http://www.johndcook.com/blog/standard_deviation/
    """
    def __init__(self, shape):
        """
        Create a running status (mean, var, std) estimator.

        Note:
            Running on CPU. Input tensors are also required to locate
            on CPU.

        Args:
            shape: Shape of input elements.
        """
        # number of pushed samples
        self._n = 0
        # mean
        self._M = t.zeros(shape, dtype=t.float64)
        # variance * (_n - 1)
        self._S = t.zeros(shape, dtype=t.float64)

    def copy(self):
        """
        Returns a copy of the running status estimator.
        """
        other = RunningStat(self.shape)
        other._n = self._n
        other._M = self._M.clone()
        other._S = self._S.clone()
        return other

    def push(self, x: t.Tensor):
        """
        Add a new sample to the running status estimator.

        Args:
            x: New sample.
        """
        assert x.shape == self._M.shape, "Shape mismatch!"
        n_old = self._n
        self._n += 1
        if self._n == 1:
            self._M = x
        else:
            delta = x - self._M
            self._M += delta / self._n
            self._S += delta * delta * n_old / self._n

    def update(self, other: "RunningStat"):
        """
        combine this estimator with another estimator.

        Args:
            other: Another running status estimator, with the same shape.
        """
        # combined variance:
        # https://www.emathzone.com/tutorials/basic-statistics/
        # combined-variance.html

        # Note: S is variance multiplied by n - 1. you need to
        # deduce a little bit to get the calculation method of S
        assert other.shape == self.shape, "Shape mismatch!"
        n1 = self._n
        n2 = other._n
        n = n1 + n2
        delta = self._M - other._M
        delta2 = delta * delta
        M = (n1 * self._M + n2 * other._M) / n
        S = self._S + other._S + delta2 * n1 * n2 / n
        self._n = n
        self._M = M
        self._S = S

    def __repr__(self):
        return 'RunningStat(shape={}, n={}, mean_mean={}, mean_std={})'.format(
            self._M.shape, self.n, t.mean(self.mean), t.mean(self.std)
        )

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        # at self._N = 0
        # return self._M instead of self._S
        # to prevent division by 0 in state normalization
        return self._S / (self._n - 1) if self._n > 1 else t.square(self._M)

    @property
    def std(self):
        return t.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class MeanStdFilter(object):
    """Keeps track of a running mean for seen states"""

    def __init__(self, shape):
        """
        Create a state normalization filter.

        Note:
            Running on CPU. Input tensors are also required to locate
            on CPU.

        Args:
            shape: Shape of the state
        """
        self.shape = shape
        self.rs = RunningStat(shape)

        # In distributed rollouts, each worker sees different states.
        # `rs_local` is used to keep track of encountered local states.
        # In each rollout episode, the manager process will collect `rs_local`
        # from all workers, then distribute new running states to `rs` of all
        # workers.

        self.rs_local = RunningStat(shape)

        self.mean = t.zeros(shape, dtype=t.float64)
        self.std = t.ones(shape, dtype=t.float64)

    def clear_local(self):
        self.rs_local = RunningStat(self.shape)
        return

    def copy(self):
        """
        Returns a copy of Filter.
        """
        other = MeanStdFilter(self.shape)
        other.rs = self.rs.copy()
        other.rs_local = self.rs_local.copy()
        return other

    def collect(self, other: "MeanStdFilter"):
        """
        Takes a worker's filter and collect their local status.

        Using notation `F(state, buffer)`, given `Filter1(x1, y1)` and
        `Filter2(x2, yt)`, `Filter1` to `Filter1(x1 + yt, y1)`.

        This function is **only** used by the manager process.
        """
        self.rs.update(other.rs_local)
        return

    def sync(self, other):
        """
        Syncs all fields together from other filter.

        Using notation `F(state, buffer)` Given `Filter1(x1, y1)` and
        `Filter2(x2, yt)`, `sync` modifies `Filter1` to `Filter1(x2, yt)`

        This function is used by all worker processes.
        """
        assert other.shape == self.shape, "Shape mismatch!"
        self.rs = other.rs.copy()
        self.rs_local = other.rs_local.copy()
        return

    def apply_stats(self):
        """
        Apply the mean and std stored in `self.rs` to normalization.
        """
        self.mean = self.rs.mean
        self.std = self.rs.std

        # Set values for std less than 1e-7 to +inf to avoid
        # dividing by zero. State elements with zero variance
        # are set to zero as a result.
        self.std[self.std < 1e-7] = float("inf")
        return

    def filter(self, x: t.Tensor, update: bool = True):
        """
        Filter(normalize) observed state.

        Note:
            `update` will only update internal records of mean and std
            of the filter stored in `self.rs` and `self.rs_local`,
            but will not update the used mean and std in normalization.

            You must call `apply_stats()` to apply mean and std from
            `self.rs`.

        Args:
            x: State to filter.
            update: Whether update the filter with the current state.

        Returns:
            Normalized state.
        """
        if update:
            if len(x.shape) == len(self.rs.shape) + 1:
                # The vectorized case.
                for i in range(x.shape[0]):
                    self.rs.push(x[i])
                    self.rs_local.push(x[i])
            else:
                # The unvectorized case.
                self.rs.push(x)
                self.rs_local.push(x)
            x = x - self.mean
            x = x / (self.std + 1e-8)
        return x

    def __repr__(self):
        return "MeanStdFilter(shape={}, rs={}, rs_local={})".format(
            self.shape, self.rs, self.rs_local
        )


class SharedNoiseSampler(object):
    def __init__(self, noise: t.Tensor, seed: int):
        """
        Args:
            noise: Noise tensor to sample from.
            seed: Index random sampling seed.
        """
        self.rg = np.random.RandomState(seed)
        self.noise = noise
        assert self.noise.dtype == np.float64

    def get(self, idx, size):
        return self.noise[idx:idx + size]

    def sample(self, size):
        """
        Args:
            size: Size of sampled 1D noise tensor.

        Returns:
            Noise begin index, noise tensor.
        """
        idx = self.rg.randint(0, len(self.noise) - size + 1)
        return idx, self.noise[idx:idx + size]


class ARS(TorchFramework):
    """
    ARS framework.
    """

    _is_top = ["actor"]
    _is_restorable = ["actor"]

    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 ars_group: RpcGroup,
                 model_server: Tuple[PushPullModelServer],
                 *_,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple] = None,
                 lr_scheduler_kwargs: Tuple[Dict] = None,
                 actor_learning_rate: float = 0.001,
                 gradient_max: float = np.inf,
                 noise_std_dev: float = 0.02,
                 noise_size: int = 250000000,
                 rollout_num: int = 32,
                 used_rollout_num: int = 32,
                 normalize_state: bool = True,
                 noise_seed: int = 12345,
                 sample_seed: int = 123,
                 **__):
        """

        Note:
            The first process in `ars_group` will be the manager process.

        Args:
            actor:
            optimizer:
            ars_group:
            model_server:
            *_:
            lr_scheduler:
            lr_scheduler_args:
            lr_scheduler_kwargs:
            actor_learning_rate:
            gradient_max:
            noise_std_dev:
            noise_size:
            rollout_num:
            used_rollout_num:
            normalize_state:
            noise_seed:
            sample_seed:
            **__:
        """
        assert rollout_num >= used_rollout_num
        self.grad_max = gradient_max
        self.rollout_num = rollout_num
        self.used_rollout_num = used_rollout_num
        self.normalize_state = normalize_state
        self.ars_group = ars_group

        self.actor = actor
        self.actor_positive = copy.deepcopy(actor)
        self.actor_negative = copy.deepcopy(actor)
        self.actor_optim = optimizer(self.actor.parameters(),
                                     lr=actor_learning_rate)
        self.actor_model_server = model_server[0]
        self.filter = {}   # type: Dict[str, MeanStdFilter]
        self.delta_idx = {}    # type: Dict[str, int]
        self.reward = [[], []]

        if lr_scheduler is not None:
            if lr_scheduler_args is None:
                lr_scheduler_args = (())
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = ({})
            self.actor_lr_sch = lr_scheduler(
                self.actor_optim,
                *lr_scheduler_args[0],
                **lr_scheduler_kwargs[0],
            )

        # generate shared noise
        # estimate model parameter num first
        param_max_num = 0
        for param in actor.parameters():
            param_max_num = max(np.prod(np.array(param.shape)), param_max_num)
        if param_max_num * 10 > noise_size:
            default_logger.warning("Maximum parameter size of your model is "
                                   "{}, which is more than 1/10 of your noise"
                                   "size {}, consider increasing noise_size."
                                   .format(param_max_num, noise_size))
        elif param_max_num >= noise_size:
            raise ValueError("Noise size {} is too small compared to"
                             "maximum parameter size {}!"
                             .format(noise_size, param_max_num))

        # create shared noise array
        noise_array = t.tensor(np.random.RandomState(noise_seed)
                               .randn(noise_size).astype(np.float64)
                               * noise_std_dev)

        # create a sampler for each parameter in model
        self.noise_sampler = {}
        process_index = (ars_group.get_group_members()
                                  .index(ars_group.get_cur_name()))
        param_num = len(list(actor.parameters()))
        for idx, (name, param) in enumerate(actor.named_parameters()):
            self.noise_sampler[name] = SharedNoiseSampler(
                noise_array, sample_seed + process_index * param_num + idx
            )

        # synchronize base actor parameters
        self._sync_parameter()
        self._generate_parameter()
        super(ARS, self).__init__()

    def act(self,
            state: Dict[str, Any],
            param_type: str,
            *_, **__):
        # normalize states
        # filter shapes will be initialized on first call
        if self.normalize_state:
            for k, v in state.items():
                if k not in self.filter:
                    self.filter[k] = MeanStdFilter(v.shape)
                state[k] = self.filter[k].filter(v.to("cpu")).to(v.device)
        if param_type == "original":
            return safe_return(safe_call(self.actor, state))
        elif param_type == "positive":
            return safe_return(safe_call(self.actor_positive, state))
        elif param_type == "negative":
            return safe_return(safe_call(self.actor_negative, state))
        else:
            raise ValueError('Invalid parameter type: {}, '
                             'available options are: '
                             '"original", "positive", "negative"'
                             .format(param_type))

    def store_reward(self,
                     reward: float,
                     param_type: str,
                     *_, **__):
        if param_type == "positive":
            self.reward[0].append(reward)
        elif param_type == "negative":
            self.reward[1].append(reward)
        else:
            raise ValueError('Invalid parameter type: {}, '
                             'available options are: '
                             '"positive", "negative"'
                             .format(param_type))

    def update(self):
        is_manager = self.ars_group.get_group_members()[0] == \
                     self.ars_group.get_cur_name()
        # calculate average reward of collected episodes
        pos_reward = np.mean(self.reward[0])
        neg_reward = np.mean(self.reward[1])

        # collect result in manager process
        self.ars_group.pair("ars/rollout_result/{}"
                            .format(self.ars_group.get_cur_name()),
                            [self.delta_idx, pos_reward, neg_reward])
        self.ars_group.pair("ars/filter/{}"
                            .format(self.ars_group.get_cur_name()),
                            self.filter)

        self.ars_group.barrier()

        if is_manager:
            delta_idxs = []  # type: List[Dict[str, int]]
            pos_rewards = []   # type: List[int]
            neg_rewards = []   # type: List[int]
            for m in self.ars_group.get_group_members():
                delta_idx, pos_reward, neg_reward = \
                    self.ars_group.get_paired("ars/rollout_result/" + m)\
                        .to_here()
                delta_idxs.append(delta_idx)
                pos_rewards.append(pos_reward)
                neg_rewards.append(neg_reward)

            # shape: [rollout_num, 2]
            rollout_rewards = np.array([pos_rewards, neg_rewards])
            max_rewards = np.max(rollout_rewards, axis=1)

            # select top performing directions if
            # used_rollout_num < rollout_num
            idx = np.arange(max_rewards.size)[
                max_rewards >= np.percentile(
                    (max_rewards,
                     100 * (1 - (self.used_rollout_num / self.rollout_num)))
                )
            ]
            delta_idxs = [delta_idxs[i] for i in idx]
            rollout_rewards = rollout_rewards[idx, :]

            # normalize rewards by their standard deviation
            rollout_rewards /= np.std(rollout_rewards)

            self.actor.zero_grad()
            # aggregate rollouts to form g, the gradient
            self._cal_gradient(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                               delta_idxs)
            # apply gradients
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_max
            )
            self.actor_optim.step()

            # collect state statistics
            filters = []  # type: List[Dict[str, MeanStdFilter]]
            for m in self.ars_group.get_group_members():
                filters.append(self.ars_group.get_paired("ars/filter/" + m)
                               .to_here())
            for k, f in self.filter:
                for ff in filters:
                    self.filter[k].collect(ff[k])
                self.filter[k].apply_stats()
                self.filter[k].clear_local()

        # synchronize filter states across all workers (and the manager)
        self._sync_filter()

        # synchronize parameters across all workers (and the manager)
        self._sync_parameter()
        self.ars_group.unpair("ars/rollout_result/{}"
                              .format(self.ars_group.get_cur_name()))
        self.ars_group.unpair("ars/filter/{}"
                              .format(self.ars_group.get_cur_name()))
        self.ars_group.barrier()

    def _cal_gradient(self,
                      reward_diff: np.array,
                      delta_idxs: List[Dict[str, int]]):
        for name, param in self.actor.parameters():
            deltas = [self.noise_sampler[name].get(delta_idx[name],
                                                   param.nelement()) * r_diff
                      for r_diff, delta_idx in zip(reward_diff, delta_idxs)]
            delta = t.mean(t.stack(deltas).to(param.device), dim=0)\
                .reshape(param.shape)
            with t.no_grad():
                param.grad = delta

    def _sync_filter(self):
        manager = self.ars_group.get_group_members()[0]
        is_manager = manager == self.ars_group.get_cur_name()

        if not is_manager:
            manager_filter = self.ars_group.get_paired("ars/filter/"
                                                       + manager).to_here()
            for k, v in self.filter:
                v.sync(manager_filter[k])
                v.apply_stats()
        self.ars_group.barrier()

    def _sync_parameter(self):
        is_manager = self.ars_group.get_group_members()[0] == \
                     self.ars_group.get_cur_name()
        if is_manager:
            self.actor_model_server.push(self.actor)
        self.ars_group.barrier()
        if not is_manager:
            self.actor_model_server.pull(self.actor)
        self.ars_group.barrier()

    def _generate_parameter(self):
        """
        Generate new actor parameters with positive and negative noise deltas.
        """
        for (name, param), (_, param_p), (__, param_n) in \
            zip(self.actor.named_parameters(),
                self.actor_positive.named_parameters(),
                self.actor_negative.named_parameters()):
            param_size = np.prod(np.array(param.shape))
            sampler = self.noise_sampler[name]
            idx, delta = sampler.sample(param_size)
            delta = delta.reshape(param.shape)
            self.delta_idx[name] = idx
            with t.no_grad():
                param_p.set_(param.data + delta.to(param.device))
                param_n.set_(param.data - delta.to(param.device))
