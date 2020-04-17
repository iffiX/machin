import os
import re
import torch
import shutil

def prep_create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    """
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)


def prep_dir_default(root_dir):
    if not os.path.exists(root_dir + "/model"):
        os.mkdir(root_dir + "/model")
    if not os.path.exists(root_dir + "/log"):
        os.mkdir(root_dir + "/log")
    if not os.path.exists(root_dir + "/log/train_log"):
        os.mkdir(root_dir + "/log/train_log")
    else:
        shutil.rmtree(root_dir + "/log/train_log")
        os.mkdir(root_dir + "/log/train_log")
    if not os.path.exists(root_dir + "/log/images"):
        os.mkdir(root_dir + "/log/images")


def prep_load_model(model_dir, network_map, version=-1, quiet=False):
    if not os.path.exists(model_dir):
        raise RuntimeError("Model directory doesn't exist!")
    version_map = {}
    for net_name in network_map.keys():
        version_map[net_name] = set()
    models = os.listdir(model_dir)
    for m in models:
        match = re.fullmatch("([a-zA-Z0-9_-]+)_([0-9]+)\.pt$", m)
        if match is not None:
            n = match.group(1)
            v = int(match.group(2))
            if n in network_map:
                version_map[n].add(v)
    if version > 0:
        is_version_found = [version in version_map[name] for name in network_map.keys()]
        if all(is_version_found):
            print("Specified version found, using version: {}".format(version))
            for net_name, net in network_map.items():
                net.load_state_dict(torch.load(model_dir + "/{}_{}.pt".format(net_name, version)))
            return
        else:
            for ivf, net_name in zip(is_version_found, network_map.keys()):
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
    for net_name, net in network_map.items():
        net.load_state_dict(torch.load(model_dir + "/{}_{}.pt".format(net_name, version)))