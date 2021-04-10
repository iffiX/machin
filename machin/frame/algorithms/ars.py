# Adapted from https://github.com/modestyachts/ARS

from typing import Union, Dict, List, Tuple, Callable, Any
import copy
import torch as t
import torch.nn as nn
import numpy as np

from machin.model.nets.base import NeuralNetworkModule
from machin.parallel.server import PushPullModelServer
from machin.parallel.distributed import RpcGroup, get_world
from machin.frame.helpers.servers import model_server_helper
from machin.utils.logging import default_logger
from .base import TorchFramework, Config
from .utils import (
    safe_call,
    safe_return,
    assert_and_get_valid_models,
    assert_and_get_valid_optimizer,
    assert_and_get_valid_lr_scheduler,
)


class RunningStat:
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
        assert x.dtype == t.float64, "Input are required to be float64!"
        assert x.shape == self._M.shape, "Shape mismatch!"
        assert x.device == self._M.device, "Device mismatch!"
        n_old = self._n
        self._n += 1
        if self._n == 1:
            self._M.copy_(x)
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
        return (
            f"RunningStat(shape={self._M.shape}, n={self.n}, "
            f"mean_mean={t.mean(self.mean)}, mean_std={t.mean(self.std)})"
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


class MeanStdFilter:
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
            self.rs.push(x)
            self.rs_local.push(x)
        x = x - self.mean
        x = x / (self.std + 1e-8)
        return x

    def __repr__(self):
        return (
            f"MeanStdFilter(shape={self.shape}, rs={self.rs}, "
            f"rs_local={self.rs_local})"
        )


class SharedNoiseSampler:
    def __init__(self, noise: t.Tensor, seed: int):
        """
        Args:
            noise: Noise tensor to sample from.
            seed: Index random sampling seed.
        """
        self.rg = np.random.RandomState(seed)
        self.noise = noise
        assert self.noise.dtype == t.float64

    def get(self, idx, size):
        return self.noise[idx : idx + size]

    def sample(self, size):
        """
        Args:
            size: Size of sampled 1D noise tensor.

        Returns:
            Noise begin index, noise tensor.
        """
        idx = self.rg.randint(0, len(self.noise) - size + 1)
        return idx, self.noise[idx : idx + size]


class ARS(TorchFramework):
    """
    ARS framework.
    """

    _is_top = ["actor"]
    _is_restorable = ["actor"]

    def __init__(
        self,
        actor: Union[NeuralNetworkModule, nn.Module],
        optimizer: Callable,
        ars_group: RpcGroup,
        model_server: Tuple[PushPullModelServer],
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple] = None,
        lr_scheduler_kwargs: Tuple[Dict] = None,
        learning_rate: float = 0.01,
        gradient_max: float = np.inf,
        noise_std_dev: float = 0.02,
        noise_size: int = 250000000,
        rollout_num: int = 32,
        used_rollout_num: int = 32,
        normalize_state: bool = True,
        noise_seed: int = 12345,
        sample_seed: int = 123,
        **__,
    ):
        """

        Note:
            The first process in `ars_group` will be the manager process.

        Args:
            actor: Actor network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            ars_group: Group of all processes using the ARS framework.
            model_server: Custom model sync server accessor for ``actor``.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            gradient_max: Maximum gradient.
            noise_std_dev: Standard deviation of the shared noise array.
            noise_size: Size of the shared noise array.
            rollout_num: Number of rollouts executed by workers in group.
            used_rollout_num: Number of used rollouts.
            normalize_state:  Whether to normalize the state seen by actor.
            noise_seed: Random seed used to generate noise.
            sample_seed: Based random seed used to sample noise.
        """
        assert rollout_num >= used_rollout_num
        self.grad_max = gradient_max
        self.rollout_num = rollout_num
        self.used_rollout_num = used_rollout_num
        self.normalize_state = normalize_state
        self.ars_group = ars_group

        # determine the number of rollouts(pair of actors with neg/pos delta)
        # assigned to current worker process
        w_num = len(ars_group.get_group_members())
        w_index = ars_group.get_group_members().index(ars_group.get_cur_name())
        segment_length = int(np.ceil(rollout_num / w_num))
        self.local_rollout_min = w_index * segment_length
        self.local_rollout_num = min(
            segment_length, rollout_num - self.local_rollout_min
        )

        self.actor = actor
        # `actor_with_delta` use rollout index and delta sign as key.
        # where rollout index is the absolute global index of rollout
        # and delta sign is true for positive, false for negative
        self.actor_with_delta = {}  # type: Dict[Tuple[int, bool], t.nn.Module]
        self.actor_optim = optimizer(self.actor.parameters(), lr=learning_rate)
        self.actor_model_server = model_server[0]

        # `filter` use state name as key
        # eg: "state_1"
        self.filter = {}  # type: Dict[str, MeanStdFilter]

        # `delta_idx` use rollout index as key
        # The inner dict use model parameter name as key, and starting
        # noise index in the noise array as value.
        self.delta_idx = {}  # type: Dict[int, Dict[str, int]]

        # `reward` use rollout index as key, the first list stores
        # rewards of model with negative noise delta, the second list
        # stores rewards of model with positive noise delta.
        self.reward = {}  # type: Dict[int, Tuple[List, List]]

        if lr_scheduler is not None:
            if lr_scheduler_args is None:
                lr_scheduler_args = ((),)
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = ({},)
            self.actor_lr_sch = lr_scheduler(
                self.actor_optim, *lr_scheduler_args[0], **lr_scheduler_kwargs[0],
            )

        # generate shared noise
        # estimate model parameter num first
        param_max_num = 0
        for param in actor.parameters():
            param_max_num = max(np.prod(np.array(param.shape)), param_max_num)
        if param_max_num * 10 > noise_size:
            default_logger.warning(
                "Maximum parameter size of your model is "
                f"{param_max_num}, which is more than 1/10 of your noise"
                f"size {noise_size}, consider increasing noise_size."
            )
        elif param_max_num >= noise_size:
            raise ValueError(
                f"Noise size {noise_size} is too small compared to"
                f"maximum parameter size {param_max_num}!"
            )

        # create shared noise array
        self.noise_array = t.tensor(
            np.random.RandomState(noise_seed).randn(noise_size).astype(np.float64)
            * noise_std_dev
        )

        # create a sampler for each parameter in each rollout model
        # key is model parameter name
        self.noise_sampler = {}  # type: Dict[int, Dict[str, SharedNoiseSampler]]
        param_num = len(list(actor.parameters()))
        for lrn in range(self.local_rollout_num):
            r_idx = lrn + self.local_rollout_min
            sampler = {}
            for p_idx, (name, param) in enumerate(actor.named_parameters()):
                # each model and its inner parameters use a different
                # sampling stream of the same noise array.
                sampler[name] = SharedNoiseSampler(
                    self.noise_array, sample_seed + r_idx * param_num + p_idx
                )
            self.noise_sampler[r_idx] = sampler

        # synchronize base actor parameters
        self._sync_actor()
        self._generate_parameter()
        self._reset_reward_dict()
        super().__init__()

    @property
    def optimizers(self):
        return [self.actor_optim]

    @optimizers.setter
    def optimizers(self, optimizers):
        self.actor_optim = optimizers[0]

    @property
    def lr_schedulers(self):
        if hasattr(self, "actor_lr_sch"):
            return [self.actor_lr_sch]
        return []

    @classmethod
    def is_distributed(cls):
        return True

    def get_actor_types(self) -> List[str]:
        """
        Returns:
            A list of actor types needed to be evaluated by current worker
            process.
        """
        names = [
            "positive_" + str(k[0]) if k[1] else "negative_" + str(k[0])
            for k in self.actor_with_delta.keys()
        ]
        return names

    def act(self, state: Dict[str, Any], actor_type: str, *_, **__):
        """
        Use actor network to give a policy to the current state.

        Args:
            state: State dict seen by actor.
            actor_type: Type of the used actor.

        Returns:
            Anything produced by actor.
        """
        # normalize states
        # filter shapes will be initialized on first call
        if self.normalize_state:
            for k, v in state.items():
                if k not in self.filter:
                    self.filter[k] = MeanStdFilter(v.shape)
                state[k] = (
                    self.filter[k]
                    .filter(v.to(dtype=t.float64, device="cpu"))
                    .to(dtype=v.dtype, device=v.device)
                )
        if actor_type == "original":
            return safe_return(safe_call(self.actor, state))
        elif actor_type.startswith("positive_") or actor_type.startswith("negative_"):
            rollout_idx = int(actor_type.split("_")[1])
            is_positive = actor_type[0] == "p"
            return safe_return(
                safe_call(self.actor_with_delta[(rollout_idx, is_positive)], state)
            )
        else:
            avail_actor_types = '", "'.join(self.get_actor_types())
            raise ValueError(
                f"Invalid parameter type: {actor_type}, "
                f'available options are: "original", "{avail_actor_types}"'
            )

    def store_reward(self, reward: float, actor_type: str, *_, **__):
        """
        Store rollout reward (usually value of the whole rollout episode) for
        each actor type.

        Args:
            reward: Rollout reward.
            actor_type: Actor type.
        """
        if actor_type.startswith("positive_") or actor_type.startswith("negative_"):
            rollout_idx = int(actor_type.split("_")[1])
            is_positive = actor_type[0] == "p"
            self.reward[rollout_idx][is_positive].append(reward)
        else:
            avail_actor_types = '", "'.join(self.get_actor_types())
            raise ValueError(
                f"Invalid parameter type: {actor_type}, "
                f'available options are: "original", "{avail_actor_types}"'
            )

    def update(self):
        """
        Update actor network using rollouts.

        Note:
            All processes in the ARS group must enter this function.
        """
        is_manager = (
            self.ars_group.get_group_members()[0] == self.ars_group.get_cur_name()
        )
        # calculate average reward of collected episodes
        pos_reward, neg_reward, delta_idx = self._get_reward_and_delta()

        # collect result in manager process
        self.ars_group.pair(
            f"ars/rollout_result/{self.ars_group.get_cur_name()}",
            [pos_reward, neg_reward, delta_idx],
        )
        if self.normalize_state:
            self.ars_group.pair(
                f"ars/filter/{self.ars_group.get_cur_name()}", self.filter
            )
        self.ars_group.barrier()

        if is_manager:
            delta_idxs = []  # type: List[Dict[str, int]]
            pos_rewards = []  # type: List[int]
            neg_rewards = []  # type: List[int]
            for m in self.ars_group.get_group_members():
                pos_reward, neg_reward, delta_idx = self.ars_group.get_paired(
                    "ars/rollout_result/" + m
                ).to_here()

                delta_idxs += delta_idx
                pos_rewards += pos_reward
                neg_rewards += neg_reward

            # shape: [2, rollout_num]
            rollout_rewards = np.array([pos_rewards, neg_rewards])
            max_rewards = np.max(rollout_rewards, axis=0)

            # select top performing directions if
            # used_rollout_num < rollout_num
            idx = np.arange(max_rewards.size)[
                max_rewards
                >= np.percentile(
                    max_rewards, 100 * (1 - (self.used_rollout_num / self.rollout_num))
                )
            ]
            delta_idxs = [delta_idxs[i] for i in idx]
            rollout_rewards = rollout_rewards[:, idx]

            # normalize rewards by their standard deviation
            var = np.std(rollout_rewards)
            if not np.isclose(var, 0.0):
                rollout_rewards /= var
            self.actor.zero_grad()
            # aggregate rollouts to form gradient
            # use neg_rollout_rewards - pos_rollout_rewards
            # because -alpha * gradient is added to
            # parameters in SGD
            self._cal_gradient(rollout_rewards[1] - rollout_rewards[0], delta_idxs)
            # nn.utils.clip_grad_norm_(
            #     self.actor.parameters(), self.grad_max
            # )
            # apply gradients
            self.actor_optim.step()

            # collect state statistics
            if self.normalize_state:
                filters = []  # type: List[Dict[str, MeanStdFilter]]
                for m in self.ars_group.get_group_members():
                    filters.append(
                        self.ars_group.get_paired("ars/filter/" + m).to_here()
                    )
                for k, f in self.filter.items():
                    for ff in filters:
                        self.filter[k].collect(ff[k])
                    self.filter[k].apply_stats()
                    self.filter[k].clear_local()

        self.ars_group.barrier()
        self.ars_group.unpair(f"ars/rollout_result/{self.ars_group.get_cur_name()}")
        if self.normalize_state:
            self.ars_group.unpair(f"ars/filter/{self.ars_group.get_cur_name()}")
        self.ars_group.barrier()

        # synchronize filter states across all workers (and the manager)
        if self.normalize_state:
            self._sync_filter()

        # synchronize parameters across all workers (and the manager)
        self._sync_actor()
        # generate new actor parameters with positive and negative delta
        self._generate_parameter()

        # reset reward dict
        self._reset_reward_dict()

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()

    def _get_reward_and_delta(self):
        r_range = [i + self.local_rollout_min for i in range(self.local_rollout_num)]
        pos_reward = []
        neg_reward = []
        delta_idx = []
        for i in r_range:
            assert self.reward[i][0] and self.reward[i][1], (
                "You must store rewards for parameters with positive "
                "noise delta and negative noise delta!"
            )
            pos_reward.append(np.mean(self.reward[i][1]))
            neg_reward.append(np.mean(self.reward[i][0]))
            delta_idx.append(self.delta_idx[i])
        return pos_reward, neg_reward, delta_idx

    def _cal_gradient(self, reward_diff: np.array, delta_idxs: List[Dict[str, int]]):
        sampler = SharedNoiseSampler(self.noise_array, 0)
        for name, param in self.actor.named_parameters():
            deltas = [
                sampler.get(delta_idx[name], param.nelement()).reshape(param.shape)
                * r_diff
                for r_diff, delta_idx in zip(reward_diff, delta_idxs)
            ]
            delta = t.mean(
                t.stack(deltas).to(dtype=param.dtype, device=param.device), dim=0
            )

            with t.no_grad():
                param.grad = delta

    def _sync_filter(self):
        is_manager = (
            self.ars_group.get_group_members()[0] == self.ars_group.get_cur_name()
        )
        if is_manager:
            self.ars_group.pair("ars/filter_m", self.filter)
        self.ars_group.barrier()
        if not is_manager:
            manager_filter = self.ars_group.get_paired("ars/filter_m").to_here()
            for k, f in self.filter.items():
                f.sync(manager_filter[k])
                f.apply_stats()
        self.ars_group.barrier()
        if is_manager:
            self.ars_group.unpair("ars/filter_m")
        self.ars_group.barrier()

    def _sync_actor(self):
        is_manager = (
            self.ars_group.get_group_members()[0] == self.ars_group.get_cur_name()
        )
        if is_manager:
            assert self.actor_model_server.push(self.actor), "Push failed"
        self.ars_group.barrier()
        if not is_manager:
            assert self.actor_model_server.pull(self.actor), "Pull failed"
        self.ars_group.barrier()

    def _reset_reward_dict(self):
        self.reward = {}
        for lrn in range(self.local_rollout_num):
            r_idx = lrn + self.local_rollout_min
            self.reward[r_idx] = ([], [])

    def _generate_parameter(self):
        """
        Generate new actor parameters with positive and negative noise deltas.
        """
        self.actor_with_delta = {}
        for lrn in range(self.local_rollout_num):
            r_idx = lrn + self.local_rollout_min
            actor_positive = copy.deepcopy(self.actor)
            actor_negative = copy.deepcopy(self.actor)
            self.delta_idx[r_idx] = {}  # type: Dict[str, int]
            for (name, param), param_p, param_n in zip(
                self.actor.named_parameters(),
                actor_positive.parameters(),
                actor_negative.parameters(),
            ):
                param_size = param.nelement()
                param_sampler = self.noise_sampler[r_idx][name]
                idx, delta = param_sampler.sample(param_size)
                delta = delta.reshape(param.shape).to(
                    dtype=param.dtype, device=param.device
                )
                self.delta_idx[r_idx][name] = idx
                with t.no_grad():
                    param_p.data.copy_(param.data + delta)
                    param_n.data.copy_(param.data - delta)
            self.actor_with_delta[(r_idx, False)] = actor_negative
            self.actor_with_delta[(r_idx, True)] = actor_positive

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        default_values = {
            "model_server_group_name": "ars_model_server",
            "model_server_members": "all",
            "ars_group_name": "ars",
            "ars_members": "all",
            "models": ["Actor"],
            "model_args": ((),),
            "model_kwargs": ({},),
            "optimizer": "Adam",
            "lr_scheduler": None,
            "lr_scheduler_args": None,
            "lr_scheduler_kwargs": None,
            "learning_rate": 0.001,
            "gradient_max": np.inf,
            "noise_std_dev": 0.02,
            "noise_size": 250000000,
            "rollout_num": 32,
            "used_rollout_num": 32,
            "normalize_state": True,
            "noise_seed": 12345,
            "sample_seed": 123,
        }
        config = copy.deepcopy(config)
        config["frame"] = "ARS"
        if "frame_config" not in config:
            config["frame_config"] = default_values
        else:
            config["frame_config"] = {**config["frame_config"], **default_values}
        return config

    @classmethod
    def init_from_config(
        cls,
        config: Union[Dict[str, Any], Config],
        model_device: Union[str, t.device] = "cpu",
    ):
        world = get_world()
        f_config = copy.deepcopy(config["frame_config"])
        ars_group = world.create_rpc_group(
            group_name=f_config["ars_group_name"],
            members=(
                world.get_members()
                if f_config["ars_members"] == "all"
                else f_config["ars_members"]
            ),
        )

        models = assert_and_get_valid_models(f_config["models"])
        model_args = f_config["model_args"]
        model_kwargs = f_config["model_kwargs"]
        models = [
            m(*arg, **kwarg).to(model_device)
            for m, arg, kwarg in zip(models, model_args, model_kwargs)
        ]

        optimizer = assert_and_get_valid_optimizer(f_config["optimizer"])
        lr_scheduler = f_config["lr_scheduler"] and assert_and_get_valid_lr_scheduler(
            f_config["lr_scheduler"]
        )
        servers = model_server_helper(
            model_num=1,
            group_name=f_config["model_server_group_name"],
            members=f_config["model_server_members"],
        )
        del f_config["optimizer"]
        del f_config["lr_scheduler"]
        frame = cls(
            *models,
            optimizer,
            ars_group,
            servers,
            lr_scheduler=lr_scheduler,
            **f_config,
        )
        return frame
