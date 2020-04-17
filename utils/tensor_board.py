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

global_board = TensorBoard()