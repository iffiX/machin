import numpy as np
import torch as t


def convert(mem: np.ndarray):
    return t.tensor(mem.reshape(1, 128).astype(np.float32) / 255)
