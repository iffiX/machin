from typing import Dict, Iterable, Any

import os
import re
import torch as t
import torch.nn as nn

from .conf import Config, get_args, merge_config, \
    replace_config, save_config, load_config_dict


def prep_args(config: Config, save_env):
    """
    Args:
        config: :class:`.Config` object to prepare.
        save_env: :class:`.SaveEnv` instance.

    Example::

        from machin.utils.conf import Config
        from machin.utils.save_env import SaveEnv
        c = Config()

        # set some config attributes
        # restart_from_trial specifies the trial name in your root
        # directory, if it is set, then prep_args will load arguments
        # from that trial record.

        c.model_save_int = 100
        c.some_root_dir = "/some_directory"
        c.restart_from_trial = "2020_05_09_15_00_31"

        save_env = SaveEnv(c.some_root_dir,
                           restart_use_trial=c.restart_from_trial)

        # If c.restart_from_trial is not None, config will be
        # overwritten by the config entries in the saved config
        # of trial.
        # Otherwise, it is saved in the directory:
        # <c.some_root_dir>/<trial_start_time>/config
        # as a json file.
        prep_args(c, save_env)
    """
    c = config
    args = get_args()
    merge_config(c, args.conf)

    # preparations
    if c.restart_from_trial is not None:
        r = c.restart_from_trial
        replace_config(c, load_config_dict(save_env.get_trial_config_file()))
        save_env.clear_trial_train_log_dir()
        # prevent overwriting
        c.restart_from_trial = r
    else:
        save_config(c, save_env.get_trial_config_file())


def prep_clear_dirs(dirs: Iterable[str]):
    """
    Args:
         dirs: a list of directories to clear
    """
    for dir_ in dirs:
        file_list = [f for f in os.listdir(dir_)]
        for f in file_list:
            os.remove(os.path.join(dir_, f))


def prep_create_dirs(dirs: Iterable[str]):
    """
    Note: will recursively create directories.

    Args:
        dirs: a list of directories to create if these directories
            are not found.
    """
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)


def prep_dirs_default(root_dir, clear_old=False):
    """
    Create default directories (/model, /log/images, /log/train_log) in the
    given root directory.

    Args:
        root_dir: root directory.
        clear_old: whether completely removes all things in the old
            default directories.
    """
    prep_create_dirs((root_dir + "/model",
                      root_dir + "/config",
                      root_dir + "/log/images",
                      root_dir + "/log/train_log"))
    if clear_old:
        prep_clear_dirs((root_dir + "/model",
                         root_dir + "/config",
                         root_dir + "/log/images",
                         root_dir + "/log/train_log"))


def prep_load_state_dict(model: nn.Module,
                         state_dict: Any):
    """
    Automatically load a **loaded state dictionary**

    Note:
        This function handles tensor device remapping.
    """
    for name, param in model.named_parameters():
        if name not in state_dict:
            print("Warning: Key {} not found in state dict.".format(name))
        state_dict[name].to(param.device)
    model.load_state_dict(state_dict)


def prep_load_model(model_dir: str,
                    model_map: Dict[str, str],
                    version: int = -1,
                    quiet: bool = False):
    """
    Automatically find and load models.

    Args:
        model_dir: Directory to save models.
        model_map: Model saving map.
        version: Version to load, if specified, otherwise automatically
            find the latest version.
        quiet: Raise no error if no valid version could be found.
    """
    if not os.path.exists(model_dir):
        raise RuntimeError("Model directory doesn't exist!")
    version_map = {}
    for net_name in model_map.keys():
        version_map[net_name] = set()
    models = os.listdir(model_dir)
    for m in models:
        match = re.fullmatch("([a-zA-Z0-9_-]+)_([0-9]+)\\.pt$", m)
        if match is not None:
            n = match.group(1)
            v = int(match.group(2))
            if n in model_map:
                version_map[n].add(v)
    if version > 0:
        is_version_found = [version in version_map[name]
                            for name in model_map.keys()]
        if all(is_version_found):
            print("Specified version found, using version: {}".format(version))
            for net_name, net in model_map.items():
                net = net  # type: nn.Module
                state_dict = t.load(model_dir +
                                    "/{}_{}.pt".format(net_name, version),
                                    map_location="cpu")
                prep_load_state_dict(net, state_dict)
            return
        else:
            for ivf, net_name in zip(is_version_found, model_map.keys()):
                if not ivf:
                    print("Specified version {} for network {} is invalid"
                          .format(version, net_name))

    print("Begin auto find")
    # use the valid, latest model
    common = set.intersection(*version_map.values())
    if len(common) == 0:
        if not quiet:
            raise RuntimeError("Cannot find a valid version for all models!")
        else:
            return
    version = max(common)
    print("Using version: {}".format(version))
    for net_name, net in model_map.items():
        state_dict = t.load(model_dir +
                            "/{}_{}.pt".format(net_name, version),
                            map_location="cpu")
        prep_load_state_dict(net, state_dict)
