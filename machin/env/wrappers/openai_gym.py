import gym
import numpy as np
from itertools import repeat
from typing import Tuple
from ..utils.openai_gym import disable_view_window
from machin.parallel.pool import Pool
from machin.parallel import get_context
from .base import *

disable_view_window()


def _call_gym_env_method(rank, all_env_idx, method, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    hold = set(_init_envs.env_idxs[rank])
    use = hold.intersection(set(all_env_idx))
    for idx in use:
        env = _init_envs.envs[rank][idx]
        if hasattr(env, method):
            return getattr(env, method)(*args, **kwargs)


def _init_envs(rank, envs, env_idxs):
    _init_envs.envs = dict()
    _init_envs.env_idxs = dict()
    # use rank to avoid overwriting when using ThreadPool
    _init_envs.envs[rank] = {i: e for i, e in zip(env_idxs, envs)}
    _init_envs.env_idxs[rank] = env_idxs


class GymTerminationError(Exception):
    def __init__(self):
        super(GymTerminationError, self).__init__(
            "One or several environments have terminated, "
            "reset before continuing."
        )


class ParallelWrapperDummy(ParallelWrapperBase):
    """
    Dummy parallel wrapper for gym environments, implemented using for-loop.

    For debug purpose only.
    """
    def __init__(self, envs: List[gym.Env]):
        """
        Args:
            envs: List of gym environments.
        """
        super(ParallelWrapperDummy, self).__init__()
        self._envs = envs
        self._terminal = np.zeros([len(self._envs)], dtype=np.bool)

    def reset(self, idx: Union[int, List[int]] = None) -> np.ndarray:
        """
        Returns:
            Batched gym states, the first dimension is environment size.
        """
        if idx is None:
            obsrv = np.stack([e.reset() for e in self._envs])
            self._terminal = np.zeros([self.size()], dtype=np.bool)
        else:
            obsrv = []
            if np.isscalar(idx):
                idx = [idx]
            for i in idx:
                obsrv.append(self._envs[i].reset())
                self._terminal[i] = False
        return obsrv

    def step(self, action: np.ndarray, idx: Union[int, List[int]] = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Let specified environment(s) run one time step. Specified environments
        must be active and have not reached terminal states before.

        Args:
            action: Actions sent to each specified environment, the size of the
                first dimension must match the number of selected environments.
            idx: Indexes of selected environments, default is all.

        Returns:
            Stacked observation, reward, terminal, info in ``np.ndarray``
        """
        if idx is None:
            idx = list(range(self.size()))
        elif np.isscalar(idx):
            idx = [idx]

        if len(action) != len(idx):
            raise ValueError("Action number must match environment number!")
        if np.any(self._terminal[idx]):
            raise GymTerminationError()

        envs = [self._envs[i] for i in idx]
        result = [e.step(a) for e, a in zip(envs, action)]
        obsrv, reward, terminal, info = zip(*result)

        obsrv = np.stack(obsrv)
        reward = np.stack(reward)
        terminal = np.stack(terminal)
        info = np.stack(info)

        self._terminal[idx] |= terminal

        return obsrv, reward, terminal, info

    def seed(self, seed: Union[int, List[int]] = None) -> List[int]:
        """
        Set seeds for all environments.

        Args:
            seed: If seed is ``int``, the same seed will be used for all
                environments.
                If seed is ``List[int]``, it must have the same size as
                the number of all environments.
                If seed is ``None``, all environments will use the default
                seed.

        Returns:
            Actual used seed returned by all environments.
        """
        if np.isscalar(seed) or seed is None:
            seed = [seed] * self.size()
        result = []
        for e, s in zip(self._envs, seed):
            if hasattr(e, 'seed'):
                result.append(e.seed(s))
        return result

    def render(self, idx: Union[int, List[int]] = None,
               *_, **__) -> List[np.ndarray]:
        """
        Render all/specified environments.

        Args:
            idx: Indexes of selected environments, default is all.

        Returns:
            A list or rendered frames, of type ``np.ndarray`` and
            size (H, W, 3).
        """
        rendered = []
        if idx is None:
            for e in self._envs:
                if np.any(self._terminal):
                    raise GymTerminationError()
                rendered.append(e.render(mode="rgb_array"))
        else:
            if np.isscalar(idx):
                idx = [idx]
            for i in idx:
                if self._terminal[i]:
                    raise GymTerminationError()
                rendered.append(self._envs[i].render(mode="rgb_array"))
        return rendered

    def close(self) -> None:
        """
        Close all environments.
        """
        for e in self._envs:
            e.close()

    def active(self) -> List[int]:
        """
        Returns: Indexes of current active environments.
        """
        return np.arange(self.size())[~self._terminal]

    def size(self) -> int:
        """
        Returns: Number of environments.
        """
        return len(self._envs)


class ParallelWrapperPool(ParallelWrapperBase):
    """
    Parallel wrapper based on thread pool or subprocess pool for gym
    environments. Environments are passed to processes/threads in pool
    during ``__init__()``, further operations on environments will only
    use indexes of environments.
    """
    def __init__(self, envs: List[gym.Env], pool=None, pool_size=None) -> None:
        """
        Args:
            envs: List of gym environments.
            pool: A pool of workers, by default it is a subprocess pool.
            pool_size: Number of workers in the pool
        """
        super(ParallelWrapperPool, self).__init__()

        if pool is not None and pool_size is not None:
            self.pool = pool
            self.pool_size = pool_size
        else:
            ctx = get_context("spawn")
            pool = Pool(context=ctx, is_daemon=False, is_global=False)
            self.pool = pool
            self.pool_size = pool_size = pool.size()

        self.env_size = env_size = len(envs)

        # delegate environments to workers
        chunk_size = int(np.ceil(env_size / pool_size))
        env_idxs = list(range(env_size))
        env_chunks = [envs[i:i + chunk_size]
                      for i in range(0, env_size, chunk_size)]
        env_idx_chunks = [env_idxs[i:i + chunk_size]
                          for i in range(0, env_size, chunk_size)]

        # initialize the pool, store environments and their indexes
        # as a member of function _init_envs()
        self.pool.starmap(_init_envs,
                          zip(range(pool_size), env_chunks, env_idx_chunks))
        self._terminal = np.zeros([env_size], dtype=np.bool)

    def reset(self, idx: Union[int, List[int]] = None) -> np.ndarray:
        """
        Returns:
            Batched gym states, the first dimension is environment size.
        """
        env_idxs = self._select_envs(idx)
        self._terminal[env_idxs] = False
        obsrv = np.stack(
            self.pool.starmap(_call_gym_env_method,
                              zip(*self._process(env_idxs), repeat("reset")))
        )
        return obsrv

    def step(self, action: np.ndarray, idx: Union[int, List[int]] = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Let specified environment(s) run one time step. Specified environments
        must be active and have not reached terminal states before.

        Args:
            action: Actions sent to each specified environment, the size of the
                first dimension must match the number of selected environments.
            idx: Indexes of selected environments, default is all.

        Returns:
            observation, reward, terminal, info
        """
        env_idxs = self._select_envs(idx)
        if len(action) != len(env_idxs):
            raise ValueError("Action number must match environment number!")

        result = self.pool.starmap(
            _call_gym_env_method,
            zip(*self._process(env_idxs), repeat("step"), action)
        )
        obsrv, reward, terminal, info = zip(*result)

        obsrv = np.stack(obsrv)
        reward = np.stack(reward)
        terminal = np.stack(terminal)
        info = np.stack(info)

        self._terminal[env_idxs] |= terminal

        return obsrv, reward, terminal, info

    def seed(self, seed: Union[int, List[int]] = None) -> List[int]:
        """
        Set seeds for all environments.

        Args:
            seed: If seed is ``int``, the same seed will be used for all
                environments.
                If seed is ``List[int]``, it must have the same size as
                the number of all environments.
                If seed is ``None``, all environments will use the default
                seed.

        Returns:
            Actual used seed returned by all environments.
        """
        if np.isscalar(seed) or seed is None:
            seed = [seed] * self.size()
        env_idxs = self._select_envs()
        result = self.pool.starmap(
            _call_gym_env_method,
            zip(*self._process(env_idxs), repeat("seed"), seed)
        )
        return result

    def render(self, idx: Union[int, List[int]] = None,
               *args, **kwargs) -> List[np.ndarray]:
        """
        Render all/specified environments.

        Args:
            idx: Indexes of selected environments, default is all.

        Returns:
            A list or rendered frames, of type ``np.ndarray`` and size
            (H, W, 3).
        """
        env_idxs = self._select_envs(idx)
        rendered = self.pool.starmap(
            _call_gym_env_method,
            zip(*self._process(env_idxs),
                repeat("render"),
                repeat(()),
                repeat({"mode": "rgb_array"}))
        )
        return rendered

    def close(self) -> None:
        """
        Close all environments.
        """
        env_idxs = self._select_envs()
        self.pool.starmap(
            _call_gym_env_method,
            zip(*self._process(env_idxs), repeat("close"))
        )

    def active(self) -> List[int]:
        """
        Returns: Indexes of current active environments.
        """
        return np.arange(self.size())[~self._terminal]

    def size(self) -> int:
        """
        Returns: Number of environments.
        """
        return self.env_size

    def _select_envs(self, idx=None):
        if idx is None:
            idx = list(range(self.env_size))
        else:
            if np.isscalar(idx):
                idx = [idx]
        return idx

    def _process(self, idx):
        return range(self.pool_size), idx


class ParallelWrapperGroup(ParallelWrapperBase):
    # TODO: implement parallel wrapper based on process group
    def reset(self, idx: Union[int, List[int], None] = None) -> Any:
        raise NotImplementedError

    def step(self, action, idx: Union[int, List[int], None] = None) -> Any:
        raise NotImplementedError

    def seed(self, seed: Union[int, List[int], None] = None) -> List[int]:
        raise NotImplementedError

    def render(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def close(self) -> Any:
        raise NotImplementedError

    def active(self) -> List[int]:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError
