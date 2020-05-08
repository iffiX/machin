import json
import argparse
from .helper_classes import Object


class Config(Object):
    pass


def get_args():
    """
    Get arguments from the commandline
    Note: --conf a=b will set <Returned Arg Object>.conf[a] = b

    Return:
         Arg object, each attribute is an argument
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config_file',
        metavar='C',
        help='The Configuration file')
    argparser.add_argument("--conf", action="append")
    args = argparser.parse_args()

    config_dict = {}
    if args.config_file is not None:
        config_dict = load_config_cdict(args.c)

    if args.conf is not None:
        for env_str in args.conf:
            name, value = env_str.split('=')
            value = eval(value)
            config_dict[name] = value
    args.conf = config_dict
    return args


def load_config_dict(json_file):
    """
    Get configs from a json file

    Args:
        json_file: path to the json config file
    Return:
        config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict


def merge_config(conf: Config, merge_conf: dict):
    for k, v in merge_conf.items():
        conf[k] = v


def replace_config(conf: Config, replace_conf: dict):
    conf.data.clear()
    for k, v in replace_conf.items():
        conf[k] = v


def save_config(conf: Config, json_file: str):
    with open(json_file, 'w') as config_file:
        json.dump(conf.data, config_file, sort_keys=True, indent=4)

