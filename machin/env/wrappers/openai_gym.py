from itertools import repeat
from typing import Tuple, Callable
from threading import Lock
from multiprocessing import get_context
import gym
import numpy as np

from machin.parallel.exception import ExceptionWithTraceback
from machin.parallel.queue import SimpleQueue
from machin.parallel.pickle import dumps, loads

from .base import *
from ..utils.openai_gym import disable_view_window


class GymTerminationError(Exception):
    def __init__(self):
        super().__init__(
            "One or several environments have terminated, " "reset before continuing."
        )


class ParallelWrapperDummy(ParallelWrapperBase):
    """
    Dummy parallel wrapper for gym environments, implemented using for-loop.

    For debug purpose only.
    """

    def __init__(self, env_creators: List[Callable[[int], gym.Env]]):
        """
        Args:
            env_creators: List of gym environment creators, used to create
                environments, accepts a index as your environment id.
        """
        super().__init__()
        self._envs = [ec(i) for ec, i in zip(env_creators, range(len(env_creators)))]
        self._terminal = np.zeros([len(self._envs)], dtype=np.bool)

    def reset(self, idx: Union[int, List[int]] = None) -> List[object]:
        """
        Returns:
            A list of gym states.
        """
        if idx is None:
            obsrv = [e.reset() for e in self._envs]
            self._terminal = np.zeros([self.size()], dtype=np.bool)
        else:
            obsrv = []
            if np.isscalar(idx):
                idx = [idx]
            for i in idx:
                obsrv.append(self._envs[i].reset())
                self._terminal[i] = False
        return obsrv

    def step(
        self, action: Union[np.ndarray, List[Any]], idx: Union[int, List[int]] = None
    ) -> Tuple[List[object], List[float], List[bool], List[dict]]:
        """
        Let specified environment(s) run one time step. Specified environments
        must be active and have not reached terminal states before.

        Args:
            action: Actions sent to each specified environment, the size of the
                first dimension must match the number of selected environments.
            idx: Indexes of selected environments, default is all.

        Returns:
            Observation, reward, terminal, and diagnostic info.
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

        obsrv = list(obsrv)
        reward = list(reward)
        terminal = list(terminal)
        info = list(info)

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
            if hasattr(e, "seed"):
                result.append(e.seed(s))
        return result

    def render(self, idx: Union[int, List[int]] = None, *_, **__) -> List[np.ndarray]:
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

    @property
    def action_space(self) -> Any:
        # DOC INHERITED
        return self._envs[0].action_space

    @property
    def observation_space(self) -> Any:
        # DOC INHERITED
        return self._envs[0].observation_space


# noinspection PyBroadException
class ParallelWrapperSubProc(ParallelWrapperBase):
    """
    Parallel wrapper based on sub processes.
    """

    def __init__(self, env_creators: List[Callable[[int], gym.Env]]) -> None:
        """
        Args:
            env_creators: List of gym environment creators, used to create
                environments on sub process workers, accepts a index as your
                environment id.
        """
        super().__init__()
        self.workers = []

        # Some environments will hang or collapse when using fork context.
        # E.g.: in "CarRacing-v0". pyglet used by gym will have render problems.

        # In case users wants to pass tensors to environments,
        # always copy all tensors to avoid errors
        ctx = get_context("spawn")
        self.cmd_queues = [
            SimpleQueue(ctx=ctx, copy_tensor=True) for _ in range(len(env_creators))
        ]
        self.result_queue = SimpleQueue(ctx=ctx, copy_tensor=True)
        for cmd_queue, ec, env_idx in zip(
            self.cmd_queues, env_creators, range(len(env_creators))
        ):
            # enable recursive serialization to support
            # lambda & local function creators.
            self.workers.append(
                ctx.Process(
                    target=self._worker,
                    args=(
                        cmd_queue,
                        self.result_queue,
                        dumps(ec, recurse=True, copy_tensor=True),
                        env_idx,
                    ),
                )
            )

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        self.env_size = env_size = len(env_creators)
        self._cmd_lock = Lock()
        self._closed = False
        tmp_env = env_creators[0](0)
        self._action_space = tmp_env.action_space
        self._obsrv_space = tmp_env.observation_space
        tmp_env.close()
        self._terminal = np.zeros([env_size], dtype=np.bool)

    def reset(self, idx: Union[int, List[int]] = None) -> List[object]:
        """
        Returns:
            A list of gym states.
        """
        env_idxs = self._select_envs(idx)
        self._terminal[env_idxs] = False
        with self._cmd_lock:
            return self._call_gym_env_method(env_idxs, "reset")

    def step(
        self, action: Union[np.ndarray, List[Any]], idx: Union[int, List[int]] = None
    ) -> Tuple[List[object], List[float], List[bool], List[dict]]:
        """
        Let specified environment(s) run one time step. Specified environments
        must be active and have not reached terminal states before.

        Args:
            action: Actions sent to each specified environment, the size of the
                first dimension must match the number of selected environments.
            idx: Indexes of selected environments, default is all.

        Returns:
            Observation, reward, terminal, and diagnostic info.
        """
        env_idxs = self._select_envs(idx)
        if len(action) != len(env_idxs):
            raise ValueError("Action number must match environment number!")

        with self._cmd_lock:
            result = self._call_gym_env_method(
                env_idxs, "step", [(act,) for act in action]
            )

        obsrv = [r[0] for r in result]
        reward = [r[1] for r in result]
        terminal = [r[2] for r in result]
        info = [r[3] for r in result]

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
        with self._cmd_lock:
            return self._call_gym_env_method(env_idxs, "seed", [(sd,) for sd in seed])

    def render(
        self, idx: Union[int, List[int]] = None, *args, **kwargs
    ) -> List[np.ndarray]:
        """
        Render all/specified environments.

        Args:
            idx: Indexes of selected environments, default is all.

        Returns:
            A list or rendered frames, of type ``np.ndarray`` and size
            (H, W, 3).
        """
        env_idxs = self._select_envs(idx)
        with self._cmd_lock:
            return self._call_gym_env_method(
                env_idxs,
                "render",
                kwargs=list(repeat({"mode": "rgb_array"}, len(env_idxs))),
            )

    def close(self) -> None:
        """
        Close all environments, including the wrapper.
        """
        with self._cmd_lock:
            if self._closed:
                return
            self._closed = True
            env_idxs = self._select_envs()
            self._call_gym_env_method(env_idxs, "close")
            for cmd_queue in self.cmd_queues:
                cmd_queue.quick_put(None)
            for worker in self.workers:
                worker.join()

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

    @property
    def action_space(self) -> Any:
        # DOC INHERITED
        return self._action_space

    @property
    def observation_space(self) -> Any:
        # DOC INHERITED
        return self._obsrv_space

    def _select_envs(self, idx=None):
        if idx is None:
            idx = list(range(self.env_size))
        else:
            if np.isscalar(idx):
                idx = [idx]
        return idx

    def _call_gym_env_method(self, env_idxs, method, args=None, kwargs=None):
        if args is None:
            args = [() for _ in range(len(env_idxs))]
        if kwargs is None:
            kwargs = [{} for _ in range(len(env_idxs))]

        result = {}
        # Check whether any process has exited with error code:
        for worker, worker_id in zip(self.workers, range(len(self.workers))):
            if worker.exitcode is None:
                continue
            if worker.exitcode == 2:
                raise RuntimeError(f"Worker {worker_id} failed to create environment.")
            elif worker.exitcode != 0:
                raise RuntimeError(
                    f"Worker {worker_id} exited with code {worker.exitcode}."
                )

        for env_idx, i in zip(env_idxs, range(len(env_idxs))):
            self.cmd_queues[env_idx].quick_put((method, args[i], kwargs[i]))
        while len(result) < len(env_idxs):
            e_idx, success, res = self.result_queue.get()
            if success:
                result[e_idx] = res
            else:
                raise res
        return [result[e_idx] for e_idx in env_idxs]

    @staticmethod
    def _worker(
        cmd_queue: SimpleQueue, result_queue: SimpleQueue, env_creator, env_idx
    ):
        env = None
        try:
            env = loads(env_creator)(env_idx)
        except Exception:
            # Something has gone wrong during environment creation,
            # exit with error.
            exit(2)
        try:
            while True:
                try:
                    command = cmd_queue.quick_get(timeout=1e-3)
                except TimeoutError:
                    continue

                try:
                    if command is not None:
                        method, args, kwargs = command
                    else:
                        # End of all tasks signal received
                        cmd_queue.close()
                        result_queue.close()
                        break
                    result = getattr(env, method)(*args, **kwargs)
                    result_queue.put((env_idx, True, result))
                except Exception as e:
                    # Something has gone wrong during execution, serialize
                    # the exception and send it back to master.
                    result_queue.put((env_idx, False, ExceptionWithTraceback(e)))
        except KeyboardInterrupt:
            cmd_queue.close()
            result_queue.close()
