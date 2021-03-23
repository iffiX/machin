import os
import tempfile
import torch as t
import numpy as np
import pytorch_lightning as pl
from typing import Iterable, List, Dict, Union, Any, Tuple, Callable
from torch.utils.data import IterableDataset
from pytorch_lightning.loggers.base import LoggerCollection
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from machin.frame.algorithms import *
from machin.utils.media import create_video, numpy_array_to_pil_image
Scalar = Any


def is_simple_space(space):
    return type(space) in (Box, Discrete, MultiBinary, MultiDiscrete)


def is_discrete_space(space):
    return type(space) in (Discrete, MultiDiscrete)


def is_continuous_space(space):
    return type(space) in (Box, MultiBinary)


def determine_precision(models):
    dtype = set()
    for model in models:
        for k, v in model.named_parameters():
            dtype.add(v.dtype)
    dtype = list(dtype)
    if len(dtype) >= 0:
        raise RuntimeError("Multiple data types of parameters detected "
                           "in models: {}, this is currently not supported "
                           "since we need to determine the data type of your "
                           "model input from your model parameter data type."
                           .format(dtype))
    return dtype[0]


def get_loggers_as_list(module: pl.LightningModule):
    if isinstance(module.logger, LoggerCollection):
        return module.logger._logger_iterable
    else:
        return [module.logger]


def log_image(module, name, image: np.ndarray):
    for logger in get_loggers_as_list(module):
        if hasattr(logger, "log_image") and callable(logger.log_image):
            logger.log_image(name, numpy_array_to_pil_image(image))


def log_video(module, name, video_frames: List[np.ndarray]):
    # create video temp file
    _fd, path = tempfile.mkstemp(suffix=".gif")
    try:
        create_video(video_frames,
                     os.path.dirname(path),
                     os.path.basename(path))
    except Exception as e:
        print(e)
        os.remove(path)
        return

    for logger in get_loggers_as_list(module):
        if hasattr(logger, "log_artifact") and callable(logger.log_artifact):
            logger.log_artifact(path, name)
    os.remove(path)


class DatasetResult:
    def __init__(self,
                 observations: List[Dict[str, Any]] = None,
                 logs: List[Dict[str, Union[Scalar, Tuple[Scalar, str]]]]
                 = None):
        self.observations = observations or []
        self.logs = logs or []

    def add_observation(self, obs: Dict[str, Any]):
        self.observations.append(obs)

    def add_log(self, log: Dict[str, Union[Scalar, Tuple[Any, Callable]]]):
        self.logs.append(log)

    def __len__(self):
        return len(self.observations)


class RLDataset(IterableDataset):
    """
    Base class for all RL Datasets.
    """
    def __init__(self, **_kwargs):
        super(RLDataset, self).__init__()

    def __iter__(self) -> Iterable:
        return self

    def __next__(self):
        raise StopIteration()


class RLGymDiscActDataset(RLDataset):
    """
    This dataset is using a discrete action openai-gym environment.

    Notes:
        The forward method of Q networks, actor networks must accept arguments
        of default names like "action", "state", and not custom names like
        "some_action", "some_state".

        All your networks should not have custom outputs after default ones
        like action, action_log_prob, etc.

    Notes:
        The environment should accept an python int number in each step.

        The environment should output a simple observation space, dict and
        tuple are not supported. The first dimension size should be 1 as it will
        be used as the batch size, if not, it will be automatically added.

        The environment should have a finite number of acting steps.

    Args:
        frame: Algorithm framework.
        env: Gym environment instance.
        act_kwargs: Additional keyword arguments passed to act functions
            of different frameworks.
    """
    def __init__(self, frame, env, render_every_episode=100, act_kwargs=None):
        super(RLGymDiscActDataset, self).__init__()
        self.frame = frame
        self.env = env
        self.render_every_episode = render_every_episode
        self.act_kwargs = act_kwargs or {}
        self._precision = determine_precision(
            [getattr(frame, m) for m in frame._is_top]
        )
        self.counter = 0
        assert type(env.action_space) == Discrete
        assert is_simple_space(env.observation_space)

    def __next__(self):
        self.counter += 1
        result = DatasetResult()
        terminal = False
        total_reward = 0
        state = t.tensor(self.env.reset(), dtype=self._precision)
        state = state if state.shape[0] == 1 else state.unsqueeze(0)

        # manual sync and then disable syncing if framework is distributed.
        getattr(self.frame, "manual_sync", lambda: None)()
        getattr(self.frame, "set_sync", lambda x: None)(False)

        rendering = []
        while not terminal:
            if self.counter % self.render_every_episode == 0:
                rendering.append(self.env.render(mode="rgb_array"))

            with t.no_grad():
                old_state = state
                # agent model inference
                if type(self.frame) in (A2C, PPO, SAC, A3C, IMPALA):
                    action = self.frame.act(
                        {"state": old_state}, **self.act_kwargs
                    )[0]
                elif type(self.frame) in (DQN, DQNPer, DQNApex, RAINBOW):
                    action = self.frame.act_discrete_with_noise(
                        {"state": old_state}, **self.act_kwargs
                    )
                elif type(self.frame) in (DDPG, DDPGPer, HDDPG, TD3, DDPGApex):
                    action = self.frame.act_discrete_with_noise(
                        {"state": old_state}, **self.act_kwargs
                    )[0]
                elif type(self.frame) in (ARS,):
                    action = self.frame.act(
                        {"state": old_state}, **self.act_kwargs
                    )
                else:
                    raise RuntimeError("Unsupported framework: {}".format(
                        type(self.frame)
                    ))
                state, reward, terminal, info = self.env.step(action.item())
                state = t.tensor(state, dtype=self._precision)
                state = state if state.shape[0] == 1 else state.unsqueeze(0)
                total_reward += reward
                result.add_observation({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal
                })
                result.add_log(info)

        if len(rendering) > 0:
            result.add_log({"video": (rendering, log_video)})
            result.add_log({"total_reward": total_reward})

        getattr(self.frame, "set_sync", lambda x: None)(True)
        return result


class RLGymContActDataset(RLDataset):
    """
    This dataset is using a contiguous action openai-gym environment.

    Notes:
        The forward method of Q networks, actor networks must accept arguments
        of default names like "action", "state", and not custom names like
        "some_action", "some_state".

        All your networks should not have custom outputs after default ones
        like action, action_log_prob, etc.

    Notes:
        The environment should accept a numpy float array in each step.

        The environment should output a simple observation space, dict and
        tuple are not supported. The first dimension size should be 1 as it will
        be used as the batch size, if not, it will be automatically added.

        The environment should have a finite number of acting steps.

    Args:
        frame: Algorithm framework.
        env: Gym environment instance.
        act_kwargs: Additional keyword arguments passed to act functions
            of different frameworks.
    """

    def __init__(self, frame, env, render_every_episode=100, act_kwargs=None):
        super(RLGymContActDataset, self).__init__()
        self.frame = frame
        self.env = env
        self.render_every_episode = render_every_episode
        self.act_kwargs = act_kwargs or {}
        self._precision = determine_precision(
            [getattr(frame, m) for m in frame._is_top]
        )
        self.counter = 0
        assert type(env.action_space) == Box
        assert is_simple_space(env.observation_space)

    def __next__(self):
        self.counter += 1
        result = DatasetResult()
        terminal = False
        total_reward = 0
        state = t.tensor(self.env.reset(), dtype=self._precision)
        state = state if state.shape[0] == 1 else state.unsqueeze(0)

        # manual sync and then disable syncing if framework is distributed.
        getattr(self.frame, "manual_sync", lambda: None)()
        getattr(self.frame, "set_sync", lambda x: None)(False)

        rendering = []

        while not terminal:
            if self.counter % self.render_every_episode == 0:
                rendering.append(self.env.render(mode="rgb_array"))

            with t.no_grad():
                old_state = state
                # agent model inference
                if type(self.frame) in (A2C, PPO, SAC, A3C, IMPALA):
                    action = self.frame.act(
                        {"state": old_state}, **self.act_kwargs
                    )[0]
                elif type(self.frame) in (DDPG, DDPGPer, HDDPG, TD3, DDPGApex):
                    action = self.frame.act_with_noise(
                        {"state": old_state}, **self.act_kwargs
                    )[0]
                elif type(self.frame) in (ARS,):
                    action = self.frame.act(
                        {"state": old_state}, **self.act_kwargs
                    )
                else:
                    raise RuntimeError("Unsupported framework: {}".format(
                        type(self.frame)
                    ))
                state, reward, terminal, info = \
                    self.env.step(action.detach().cpu().numpy())
                state = t.tensor(state, dtype=self._precision)
                state = state if state.shape[0] == 1 else state.unsqueeze(0)
                total_reward += reward
                result.add_observation({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal
                })
                result.add_log(info)

        if len(rendering) > 0:
            result.add_log({"video": (rendering, log_video)})
            result.add_log({"total_reward": total_reward})

        getattr(self.frame, "set_sync", lambda x: None)(True)
        return result
