import torch as t
from typing import Iterable
from torch.utils.data import IterableDataset
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from machin.frame.algorithms import *


def is_simple_space(space):
    return type(space) in (Box, Discrete, MultiBinary, MultiDiscrete)


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
        tuple are not supported.
        The environment should have a finite number of acting steps.

    Args:
        algo: Algorithm framework.
        env: Gym environment instance.
        act_kwargs: Additional keyword arguments passed to act functions
            of different frameworks.
    """
    def __init__(self, algo, env, act_kwargs=None):
        super(RLGymDiscActDataset, self).__init__()
        self.algo = algo
        self.env = env
        self.act_kwargs = act_kwargs or {}
        self._precision = determine_precision(
            [getattr(algo, m) for m in algo._is_top]
        )
        assert type(env.action_space) == Discrete
        assert is_simple_space(env.observation_space)

    def __next__(self):
        observations = []
        terminal = False

        # manual sync and then disable syncing if framework is distributed.
        getattr(self.algo, "manual_sync", lambda: None)()
        getattr(self.algo, "set_sync", lambda x: None)(False)

        while not terminal:
            with t.no_grad():
                old_state = state
                # agent model inference
                if type(self.algo) in (A2C, PPO, SAC, A3C, IMPALA):
                    action = self.algo.act(
                        {"state": old_state}, **self.act_kwargs
                    )[0]
                elif type(self.algo) in (DQN, DQNPer, DQNApex, RAINBOW):
                    action = self.algo.act_discrete_with_noise(
                        {"state": old_state}, **self.act_kwargs
                    )
                elif type(self.algo) in (DDPG, DDPGPer, HDDPG, TD3):
                    action = self.algo.act_discrete_with_noise(
                        {"state": old_state}, **self.act_kwargs
                    )[0]
                elif type(self.algo) in (ARS,):
                    action = self.algo.act(
                        {"state": old_state}, **self.act_kwargs
                    )
                else:
                    raise RuntimeError("Unsupported framework: {}".format(
                        type(self.algo)
                    ))
                state, reward, terminal, info = self.env.step(action.item())
                state = t.tensor(state, dtype=self._precision).view(1, -1)

                observations.append({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal
                })

        getattr(self.algo, "set_sync", lambda x: None)(True)
        return observations


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
        tuple are not supported.
        The environment should have a finite number of acting steps.
    """

    def __init__(self, algo, env, act_kwargs=None):
        super(RLGymContActDataset, self).__init__()
        self.algo = algo
        self.env = env
        self.act_kwargs = act_kwargs or {}
        self._precision = determine_precision(
            [getattr(algo, m) for m in algo._is_top]
        )
        assert type(env.action_space) == Box
        assert is_simple_space(env.observation_space)

    def __next__(self):
        observations = []
        terminal = False

        # manual sync and then disable syncing if framework is distributed.
        getattr(self.algo, "manual_sync", lambda: None)()
        getattr(self.algo, "set_sync", lambda x: None)(False)

        while not terminal:
            with t.no_grad():
                old_state = state
                # agent model inference
                if type(self.algo) in (A2C, PPO, SAC, A3C, IMPALA):
                    action = self.algo.act(
                        {"state": old_state}, **self.act_kwargs
                    )[0]
                elif type(self.algo) in (DDPG, DDPGPer, HDDPG, TD3):
                    action = self.algo.act_with_noise(
                        {"state": old_state}, **self.act_kwargs
                    )[0]
                elif type(self.algo) in (ARS,):
                    action = self.algo.act(
                        {"state": old_state}, **self.act_kwargs
                    )
                else:
                    raise RuntimeError("Unsupported framework: {}".format(
                        type(self.algo)
                    ))
                state, reward, terminal, info = \
                    self.env.step(action.detach().cpu().numpy())
                state = t.tensor(state, dtype=self._precision).view(1, -1)

                observations.append({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal
                })

        getattr(self.algo, "set_sync", lambda x: None)(True)
        return observations
