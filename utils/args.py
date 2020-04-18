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
    return args


def set_env(args):
    if args.env is not None:
        for env_str in args.env:
            name, value = env_str.split('=')
            globals()[name] = eval(value)



