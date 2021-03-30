from typing import Dict, Iterable, Any
from os.path import join
from .logging import default_logger

import os
import re
import shutil
import torch as t
import torch.nn as nn


def prep_clear_dirs(dirs: Iterable[str]):
    """
    Args:
         dirs: a list of directories to clear
    """
    for dir_ in dirs:
        file_list = [f for f in os.listdir(dir_)]
        for f in file_list:
            f = os.path.join(dir_, f)
            if os.path.isfile(f) or os.path.islink(f):
                os.unlink(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)


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


def prep_load_state_dict(model: nn.Module, state_dict: Any):
    """
    Automatically load a **loaded state dictionary**

    Note:
        This function handles tensor device remapping.
    """
    for name, param in model.state_dict().items():
        state_dict[name].to(param.device)
    model.load_state_dict(state_dict)


def prep_load_model(
    model_dir: str,
    model_map: Dict[str, t.nn.Module],
    version: int = None,
    quiet: bool = False,
    logger: Any = None,
):
    """
    Automatically find and load models.

    Args:
        model_dir: Directory to save models.
        model_map: Model saving map.
        version: Version to load, if specified, otherwise automatically
            find the latest version.
        quiet: Raise no error if no valid version could be found.
        logger: Logger to use.
    """
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        raise RuntimeError("Model directory doesn't exist!")
    if logger is None:
        logger = default_logger

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
    if version is not None:
        is_version_found = [version in version_map[name] for name in model_map.keys()]
        if all(is_version_found):
            logger.info(f"Specified version found, using version: {version}")
            for net_name, net in model_map.items():
                net = net  # type: nn.Module
                state_dict = t.load(
                    join(model_dir, f"{net_name}_{version}.pt"), map_location="cpu",
                ).state_dict()
                prep_load_state_dict(net, state_dict)
            return
        else:
            for ivf, net_name in zip(is_version_found, model_map.keys()):
                if not ivf:
                    logger.warning(
                        f"Specified version {version} for network {net_name} is invalid"
                    )

    logger.info("Begin auto find")
    # use the valid, latest model
    common = set.intersection(*version_map.values())
    if len(common) == 0:
        if not quiet:
            raise RuntimeError("Cannot find a valid version for all models!")
        else:
            return
    version = max(common)
    logger.info(f"Using version: {version}")
    for net_name, net in model_map.items():
        state_dict = t.load(
            join(model_dir, f"{net_name}_{version}.pt"), map_location="cpu"
        ).state_dict()
        prep_load_state_dict(net, state_dict)
