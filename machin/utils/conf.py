from typing import Union
import copy
import json
import argparse

from .helper_classes import Object


class Config(Object):
    """
    A simple replacement for python dict.
    """

    def __init__(self, **configs):
        super().__init__(configs)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __iter__(self):
        for key in self.__dict__:
            if not key.startswith("__"):
                yield key

    def __contains__(self, key):
        assert not key.startswith("__")
        return hasattr(self, key)

    def __getitem__(self, key):
        assert not key.startswith("__")
        return getattr(self, key)

    def __setitem__(self, key, value):
        assert not key.startswith("__")
        setattr(self, key, value)


def load_config_cmd(merge_conf: Config = None) -> Config:
    """
    Get configs from the commandline by using "--conf".

    ``--conf a=b`` will set ``<Returned Config>.a = b``

    Example::

        python3 test.py --conf device=\"cuda:1\"
                        --conf some_dict={\"some_key\":1}

    Example::

        from machin.utils.conf import Config
        from machin.utils.save_env import SaveEnv

        # set some config attributes
        c = Config(
            model_save_int = 100,
            root_dir = "some_directory",
            restart_from_trial = "2020_05_09_15_00_31"
        )

        load_config_cmd(c)

        # restart_from_trial specifies the trial name in your root
        # directory.
        # If it is set, then SaveEnv constructor will
        # load arguments from that trial record, will overwrite.
        # If not, then SaveEnv constructor will save configurations
        # as: ``<c.some_root_dir>/<trial_start_time>/config/config.json``

        save_env = SaveEnv(c)

    Args:
        merge_conf: Config to merge.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", action="append")
    args = parser.parse_args()

    config_dict = {}
    if args.conf is not None:
        for env_str in args.conf:
            name, value = env_str.split("=")
            value = eval(value)
            config_dict[name] = value

    return merge_config((Config() if merge_conf is None else merge_conf), config_dict)


def load_config_file(json_file: str, merge_conf: Config = None) -> Config:
    """
    Get configs from a json file.

    Args:
        json_file: Path to the json config file.
        merge_conf: Config to merge.

    Return:
        configuration
    """
    # parse the configurations from the config json file provided
    with open(json_file) as config_file:
        config_dict = json.load(config_file)

    return merge_config((Config() if merge_conf is None else merge_conf), config_dict)


def save_config(conf: Config, json_file: str):
    """
    Dump config object to a json file.
    """
    with open(json_file, "w") as config_file:
        json.dump(conf.data, config_file, sort_keys=True, indent=4)


def merge_config(conf: Config, merge: Union[dict, Config]) -> Config:
    """
    Merge config object with a dictionary, or a Config object,
    same keys in the ``conf`` will be overwritten by keys
    in ``merge``.
    """
    new_conf = copy.deepcopy(conf)
    if isinstance(merge, dict):
        for k, v in merge.items():
            new_conf[k] = v
    else:
        for k, v in merge.data.items():
            if k not in new_conf.const_attrs:
                new_conf[k] = v
    return new_conf
