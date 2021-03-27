from copy import deepcopy
from typing import Dict, Any, Union
from machin.frame.algorithms import TorchFramework
from machin.utils.conf import Config
import machin.frame.algorithms as algorithms


def _fill_default(default: Union[Dict[str, Any], Config],
                  config: Union[Dict[str, Any], Config]):
    for key in default:
        if key not in config:
            config[key] = default[key]
    return config


def _get_available_algorithms():
    algos = []
    for algo in dir(algorithms):
        algo_obj = getattr(algorithms, algo)
        if issubclass(algo_obj, TorchFramework):
            algos.append(algo)
    return algos


def generate_algorithm_config(algorithm: str,
                              config: Union[Dict[str, Any], Config] = None):
    config = deepcopy(config) or {}
    if hasattr(algorithms, algorithm):
        algo_obj = getattr(algorithms, algorithm)
        if issubclass(algo_obj, TorchFramework):
            return algo_obj.generate_config(config)
    raise ValueError("Invalid algorithm: {}, valid ones are: {}"
                     .format(algorithm, _get_available_algorithms()))


def generate_gym_env_config(env_name: str = None,
                            config: Union[Dict[str, Any], Config] = None):
    """
    Generate example OpenAI gym config.
    """
    config = deepcopy(config) or {}
    return _fill_default(
        {"trials_dir": "trials",
         "gpus": 0,
         "episode_per_epoch": 100,
         "max_episodes": 1000000,
         "train_env_config": {
             "env_name": env_name or "CartPole-v1",
             "render_every_episode": 100,
             "act_kwargs": {}
         },
         "test_env_config": {
             "env_name": env_name or "CartPole-v1",
             "render_every_episode": 100,
             "act_kwargs": {}
         }},
        config
    )


def init_algorithm_from_config(config: Union[Dict[str, Any], Config]):
    assert_config_complete(config)
    frame = getattr(algorithms, config["frame"], None)
    if not isinstance(frame, TorchFramework):
        raise ValueError("Invalid algorithm: {}, valid ones are: {}"
                         .format(config["frame"], _get_available_algorithms()))
    return frame.init_from_config(config)


def is_algorithm_distributed(config: Union[Dict[str, Any], Config]):
    assert_config_complete(config)
    frame = getattr(algorithms, config["frame"], None)
    if not isinstance(frame, TorchFramework):
        raise ValueError("Invalid algorithm: {}, valid ones are: {}"
                         .format(config["frame"], _get_available_algorithms()))
    return frame.is_distributed()


def assert_config_complete(config: Union[Dict[str, Any], Config]):
    assert "frame" in config, 'Missing key "frame" in config.'
    assert "frame_config" in config, 'Missing key "frame_config" in config.'
    assert "train_env_config" in config, 'Missing key "train_env_config" ' \
                                         'in config.'
    assert "test_env_config" in config, 'Missing key "test_env_config" ' \
                                        'in config.'
