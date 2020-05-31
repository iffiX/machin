import numpy as np
from tensorboardX import SummaryWriter
from .helper_classes import Counter


class TensorBoard:
    writer = None
    counter = Counter()

    def init(self, *writer_args):
        if self.writer is None:
            self.writer = SummaryWriter(*writer_args)
        else:
            raise RuntimeError("Writer has been initialized!")

    def is_inited(self):
        return not self.writer is None


def normalize_seq_length(seq, length):
    return np.tile(np.array(seq), int(np.ceil(length / len(seq))))[:length].tolist()


global_board = TensorBoard()
