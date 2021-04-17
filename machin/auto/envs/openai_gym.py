from copy import deepcopy
from typing import Dict, List, Any, Union
from ..config import fill_default, is_algorithm_distributed
from ..pl_logger import LocalMediaLogger
from ..dataset import DatasetResult, RLDataset, log_video, determine_precision
from ..launcher import Launcher, DistributedLauncher
from machin.frame.algorithms import *
from machin.env.utils.openai_gym import disable_view_window
from machin.utils.conf import Config
from machin.utils.save_env import SaveEnv
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
import gym
import torch as t
import pytorch_lightning as pl


disable_view_window()


def _is_simple_space(space):
    return type(space) in (Box, Discrete, MultiBinary, MultiDiscrete)


def _is_discrete_space(space):
    return type(space) in (Discrete, MultiDiscrete)


def _is_continuous_space(space):
    return type(space) in (Box, MultiBinary)


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

    early_stopping_monitor = "total_reward"

    def __init__(
        self,
        frame,
        env,
        render_every_episode: int = 100,
        act_kwargs: Dict[str, Any] = None,
    ):
        super().__init__()
        self.frame = frame
        self.env = env
        self.render_every_episode = render_every_episode
        self.act_kwargs = act_kwargs or {}
        self._precision = determine_precision(
            [getattr(frame, m) for m in frame._is_top]
        )
        self.counter = 0
        assert type(env.action_space) == Discrete
        assert _is_simple_space(env.observation_space)

    def __next__(self):
        result = DatasetResult()
        terminal = False
        total_reward = 0
        state = t.tensor(self.env.reset(), dtype=self._precision)
        state = state.flatten().unsqueeze(0)

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
                    action = self.frame.act({"state": old_state}, **self.act_kwargs)[0]
                elif type(self.frame) in (DQN, DQNPer, DQNApex, RAINBOW):
                    action = self.frame.act_discrete_with_noise(
                        {"state": old_state}, **self.act_kwargs
                    )
                elif type(self.frame) in (DDPG, DDPGPer, HDDPG, TD3, DDPGApex):
                    action, probs = self.frame.act_discrete_with_noise(
                        {"state": old_state}, **self.act_kwargs
                    )
                elif type(self.frame) in (ARS,):
                    action = self.frame.act({"state": old_state}, **self.act_kwargs)
                else:
                    raise RuntimeError(f"Unsupported framework: {type(self.frame)}")
                state, reward, terminal, info = self.env.step(action.item())
                state = t.tensor(state, dtype=self._precision)
                state = state.flatten().unsqueeze(0)
                reward = float(reward)
                total_reward += reward
                if type(self.frame) in (DDPG, DDPGPer, HDDPG, TD3, DDPGApex):
                    action = probs
                result.add_observation(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal,
                    }
                )
                result.add_log(info)

        if len(rendering) > 0:
            result.add_log({"video": (rendering, log_video)})
        result.add_log({"total_reward": total_reward})

        getattr(self.frame, "set_sync", lambda x: None)(True)
        self.counter += 1
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

    early_stopping_monitor = "total_reward"

    def __init__(
        self,
        frame,
        env,
        render_every_episode: int = 100,
        act_kwargs: Dict[str, Any] = None,
    ):
        super().__init__()
        self.frame = frame
        self.env = env
        self.render_every_episode = render_every_episode
        self.act_kwargs = act_kwargs or {}
        self._precision = determine_precision(
            [getattr(frame, m) for m in frame._is_top]
        )
        self.counter = 0
        assert type(env.action_space) == Box
        assert _is_simple_space(env.observation_space)

    def __next__(self):
        result = DatasetResult()
        terminal = False
        total_reward = 0
        state = t.tensor(self.env.reset(), dtype=self._precision)
        state = state.flatten().unsqueeze(0)

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
                    action = self.frame.act({"state": old_state}, **self.act_kwargs)[0]
                elif type(self.frame) in (DDPG, DDPGPer, HDDPG, TD3, DDPGApex):
                    action = self.frame.act_with_noise(
                        {"state": old_state}, **self.act_kwargs
                    )[0]
                elif type(self.frame) in (ARS,):
                    action = self.frame.act({"state": old_state}, **self.act_kwargs)
                else:
                    raise RuntimeError(f"Unsupported framework: {type(self.frame)}")
                state, reward, terminal, info = self.env.step(
                    action.detach().cpu().numpy()
                )
                state = t.tensor(state, dtype=self._precision)
                state = state.flatten().unsqueeze(0)
                reward = float(reward)
                total_reward += reward
                result.add_observation(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal,
                    }
                )
                result.add_log(info)

        if len(rendering) > 0:
            result.add_log({"video": (rendering, log_video)})
        result.add_log({"total_reward": total_reward})

        getattr(self.frame, "set_sync", lambda x: None)(True)
        self.counter += 1
        return result


def gym_env_dataset_creator(frame, env_config):
    env = gym.make(env_config["env_name"])
    if _is_discrete_space(env.action_space):
        return RLGymDiscActDataset(
            frame,
            env,
            render_every_episode=env_config["render_every_episode"],
            act_kwargs=env_config["act_kwargs"],
        )
    elif _is_continuous_space(env.action_space):
        return RLGymContActDataset(
            frame,
            env,
            render_every_episode=env_config["render_every_episode"],
            act_kwargs=env_config["act_kwargs"],
        )
    else:
        raise ValueError(
            f"Gym environment {env_config['env_name']} has action space "
            f"of type {env.action_space}, which is not supported."
        )


def generate_env_config(
    env_name: str = None, config: Union[Dict[str, Any], Config] = None
):
    """
    Generate example OpenAI gym config.
    """
    config = deepcopy(config) or {}
    return fill_default(
        {
            "env": "openai_gym",
            "train_env_config": {
                "env_name": env_name or "CartPole-v1",
                "render_every_episode": 100,
                "act_kwargs": {},
            },
            "test_env_config": {
                "env_name": env_name or "CartPole-v1",
                "render_every_episode": 100,
                "act_kwargs": {},
            },
        },
        config,
    )


def launch(config: Union[Dict[str, Any], Config], pl_callbacks: List[Callback] = None):
    """
    Args:
        config: All configs needed to launch a gym environment and initialize
            the algorithm framework.
        pl_callbacks: Additional callbacks used to modify training behavior.

    Returns:

    """
    pl_callbacks = pl_callbacks or []

    # disable time formatting so all processes will use the same directory
    # delay directory creation
    s_env = SaveEnv(config.get("root_dir", None) or "./trial", time_format="")

    checkpoint_callback = ModelCheckpoint(
        dirpath=s_env.get_trial_model_dir(),
        filename="{epoch:02d}-{total_reward:.2f}",
        save_top_k=1,
        monitor="total_reward",
        mode="max",
        period=1,
        verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor="total_reward", mode="max", patience=config["early_stopping_patience"],
    )
    t_logger = TensorBoardLogger(s_env.get_trial_train_log_dir())
    lm_logger = LocalMediaLogger(
        s_env.get_trial_image_dir(), s_env.get_trial_image_dir()
    )

    is_distributed = is_algorithm_distributed(config)
    trainer = pl.Trainer(
        gpus=config.get("gpus", None),
        num_processes=config.get("num_processes", 1),
        num_nodes=config.get("num_nodes", 1),
        callbacks=[checkpoint_callback, early_stopping] + pl_callbacks,
        logger=[t_logger, lm_logger],
        limit_train_batches=config["episode_per_epoch"],
        max_steps=config["max_episodes"],
        accelerator="ddp" if is_distributed else None,
    )
    if is_distributed:
        model = DistributedLauncher(config, gym_env_dataset_creator)
    else:
        model = Launcher(config, gym_env_dataset_creator)
    trainer.fit(model)
