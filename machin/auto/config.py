from copy import deepcopy
from typing import Dict, Any, Union
from machin.frame.algorithms import TorchFramework
from machin.utils.conf import Config
from . import envs
import inspect
import torch as t
import machin.frame.algorithms as algorithms


def fill_default(
    default: Union[Dict[str, Any], Config], config: Union[Dict[str, Any], Config]
):
    for key in default:
        if key not in config:
            config[key] = default[key]
    return config


def get_available_algorithms():
    algos = []
    for algo in dir(algorithms):
        algo_cls = getattr(algorithms, algo)
        if (
            inspect.isclass(algo_cls)
            and issubclass(algo_cls, TorchFramework)
            and algo_cls != TorchFramework
        ):
            algos.append(algo)
    return algos


def get_available_environments():
    environments = []
    for e in dir(envs):
        e_module = getattr(envs, e)
        if hasattr(e_module, "launch") and hasattr(e_module, "generate_env_config"):
            environments.append(e)
    return environments


def generate_training_config(
    root_dir: str = "trial",
    episode_per_epoch: int = 10,
    max_episodes: int = 10000,
    config: Union[Dict[str, Any], Config] = None,
):
    config = deepcopy(config) or {}
    config["root_dir"] = root_dir
    config["episode_per_epoch"] = episode_per_epoch
    config["max_episodes"] = max_episodes
    config["early_stopping_patience"] = 3
    return config


def generate_algorithm_config(
    algorithm: str, config: Union[Dict[str, Any], Config] = None
):
    config = deepcopy(config) or {}
    if hasattr(algorithms, algorithm):
        algo_obj = getattr(algorithms, algorithm)
        if inspect.isclass(algo_obj) and issubclass(algo_obj, TorchFramework):
            config = algo_obj.generate_config(config)
            if algo_obj.is_distributed():
                # in pytorch lightning, gpus will override num_processes
                config["gpus"] = [0, 0, 0]
                config["num_processes"] = 3
                config["num_nodes"] = 1
            else:
                config["gpus"] = [0]
            return config
    raise ValueError(
        f"Invalid algorithm: {algorithm}, valid ones are: {get_available_algorithms()}"
    )


def generate_env_config(environment: str, config: Union[Dict[str, Any], Config] = None):
    config = deepcopy(config) or {}
    if hasattr(envs, environment):
        e_module = getattr(envs, environment)
        if hasattr(e_module, "launch") and hasattr(e_module, "generate_env_config"):
            return e_module.generate_env_config(config)
    raise ValueError(
        f"Invalid environment: {environment}, "
        f"valid ones are: {get_available_algorithms()}"
    )


def init_algorithm_from_config(
    config: Union[Dict[str, Any], Config], model_device: Union[str, t.device] = "cpu"
):
    assert_algorithm_config_complete(config)
    frame = getattr(algorithms, config["frame"], None)
    if not inspect.isclass(frame) or not issubclass(frame, TorchFramework):
        raise ValueError(
            f"Invalid algorithm: {config['frame']}, "
            f"valid ones are: {get_available_algorithms()}"
        )
    return frame.init_from_config(config, model_device=model_device)


def is_algorithm_distributed(config: Union[Dict[str, Any], Config]):
    assert_algorithm_config_complete(config)
    frame = getattr(algorithms, config["frame"], None)
    if not inspect.isclass(frame) or not issubclass(frame, TorchFramework):
        raise ValueError(
            f"Invalid algorithm: {config['frame']}, "
            f"valid ones are: {get_available_algorithms()}"
        )
    return frame.is_distributed()


def assert_training_config_complete(config: Union[Dict[str, Any], Config]):
    assert "root_dir" in config, 'Missing key "root_dir"'
    assert "episode_per_epoch" in config, 'Missing key "episode_per_epoch"'
    assert "max_episodes" in config, 'Missing key "max_episodes"'
    assert "early_stopping_patience" in config, 'Missing key "early_stopping_patience"'


def assert_algorithm_config_complete(config: Union[Dict[str, Any], Config]):
    assert "frame" in config, 'Missing key "frame" in config.'
    assert "frame_config" in config, 'Missing key "frame_config" in config.'


def assert_env_config_complete(config: Union[Dict[str, Any], Config]):
    assert "env" in config, 'Missing key "env" ' "in config."
    assert "train_env_config" in config, 'Missing key "train_env_config" ' "in config."
    assert "test_env_config" in config, 'Missing key "test_env_config" ' "in config."


def launch(config: Union[Dict[str, Any], Config]):
    assert_training_config_complete(config)
    assert_env_config_complete(config)
    assert_algorithm_config_complete(config)
    e_module = getattr(envs, config["env"])
    return e_module.launch(config)
