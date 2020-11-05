import numpy as np
import torch as t


def convert(mem: np.ndarray):
    """
    Convert numpy array to a tensor.

    Args:
        mem: (array): write your description
        np: (todo): write your description
        ndarray: (array): write your description
    """
    return t.tensor(mem.reshape(1, 128).astype(np.float32) / 255)
