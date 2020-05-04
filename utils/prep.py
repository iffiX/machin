import os
import re
import torch
import torch.nn


def prep_clear_dirs(dirs):
    """
    :param dirs: a list of directories to clear
    """
    for dir_ in dirs:
        file_list = [f for f in os.listdir(dir_)]
        for f in file_list:
            os.remove(os.path.join(dir_, f))


def prep_create_dirs(dirs):
    """
    :param dirs: a list of directories to create if these directories are not found
    Note: will recursively create directories.
    """
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)


def prep_dirs_default(root_dir, clear_old=False):
    """
    Create default directories (/model, /log/images, /log/train_log) in the given root directory.
    :param root_dir: root directory.
    :param clear_old: whether completely removes all things in the old default directories.
    """
    prep_create_dirs((root_dir + "/model",
                      root_dir + "/log/images",
                      root_dir + "/log/train_log"))
    if clear_old:
        prep_clear_dirs((root_dir + "/model",
                         root_dir + "/log/images",
                         root_dir + "/log/train_log"))


def prep_load_state_dict(model: torch.nn.Module, state_dict: dict):
    """
    Automatically load a ***loaded state dictionary*** this function handles tensor device remapping
    """
    for name, param in model.named_parameters():
        if name not in state_dict:
            print("Warning: Key {} not found in state dict.".format(name))
        state_dict[name].to(param.device)
    model.load_state_dict(state_dict)


def prep_load_model(model_dir, model_map, version=-1, quiet=False):
    """
    Automatically find and load models
    :param model_dir: directory to save models
    :param model_map: model saving map
    :param version: version to load, if specifier, otherwise automatically find the latest version
    :param quiet: raise no error if no valid version could be found
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
        is_version_found = [version in version_map[name] for name in model_map.keys()]
        if all(is_version_found):
            print("Specified version found, using version: {}".format(version))
            for net_name, net in model_map.items():
                state_dict = torch.load(model_dir + "/{}_{}.pt".format(net_name, version), map_location="cpu")
                prep_load_state_dict(net, state_dict)
            return
        else:
            for ivf, net_name in zip(is_version_found, model_map.keys()):
                if not ivf:
                    print("Specified version {} for network {} is invalid".format(version, net_name))

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
        state_dict = torch.load(model_dir + "/{}_{}.pt".format(net_name, version), map_location="cpu")
        prep_load_state_dict(net, state_dict)
