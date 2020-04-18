import argparse


def get_args():
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
    return args



