import json
import argparse


def get_args():
    """
    Get arguments from the commandline
    Note: --env a=b will set <Returned Arg Object>.env[a] = b

    Return:
         Arg object, each attribute is an argument
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument("--env", action="append")
    args = argparser.parse_args()
    if args.env is not None:
        env_dict = {}
        for env_str in args.env:
            name, value = env_str.split('=')
            value = eval(value)
            env_dict[name] = value
        args.env = env_dict
    else:
        args.env = {}
    return args


def get_args_from_json(json_file):
    """
    Get arguments from a json file

    Args:
        json_file: path to the json config file
    Return:
        config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict
