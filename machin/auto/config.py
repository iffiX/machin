from copy import deepcopy
from typing import Dict, Any, Union
from machin.frame.algorithms import TorchFramework
from machin.utils.conf import Config
import inspect
import machin.frame.algorithms as algorithms


def fill_default(
    default: Union[Dict[str, Any], Config], config: Union[Dict[str, Any], Config]
):
    for key in default:
        if key not in config:
            config[key] = default[key]
    return config


def _get_available_algorithms():
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


def generate_training_config(trials_dir: str = "./trials",
                             episode_per_epoch: int = 10,
                             max_episodes: int = 10000,
                             config: Union[Dict[str, Any], Config] = None):
    config = deepcopy(config) or {}
    config["trials_dir"] = trials_dir
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
        if issubclass(algo_obj, TorchFramework):
            return algo_obj.generate_config(config)
    raise ValueError(
        "Invalid algorithm: {}, valid ones are: {}".format(
            algorithm, _get_available_algorithms()
        )
    )


def init_algorithm_from_config(config: Union[Dict[str, Any], Config]):
    assert_algorithm_config_complete(config)
    frame = getattr(algorithms, config["frame"], None)
    if not inspect.isclass(frame) or not issubclass(frame, TorchFramework):
        raise ValueError(
            "Invalid algorithm: {}, valid ones are: {}".format(
                config["frame"], _get_available_algorithms()
            )
        )
    return frame.init_from_config(config)


def is_algorithm_distributed(config: Union[Dict[str, Any], Config]):
    assert_algorithm_config_complete(config)
    frame = getattr(algorithms, config["frame"], None)
    if not inspect.isclass(frame) or not issubclass(frame, TorchFramework):
        raise ValueError(
            "Invalid algorithm: {}, valid ones are: {}".format(
                config["frame"], _get_available_algorithms()
            )
        )
    return frame.is_distributed()


def assert_algorithm_config_complete(config: Union[Dict[str, Any], Config]):
    assert "frame" in config, 'Missing key "frame" in config.'
    assert "frame_config" in config, 'Missing key "frame_config" in config.'


def assert_env_config_complete(config: Union[Dict[str, Any], Config]):
    assert "train_env_config" in config, 'Missing key "train_env_config" ' "in config."
    assert "test_env_config" in config, 'Missing key "test_env_config" ' "in config."
