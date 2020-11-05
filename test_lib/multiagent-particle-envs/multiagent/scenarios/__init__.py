import imp
import os.path as osp


def load(name):
    """
    Load a python module from a file.

    Args:
        name: (str): write your description
    """
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)
