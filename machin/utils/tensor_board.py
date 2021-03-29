"""
Attributes:
    default_board: The default global board.
"""
import numpy as np
from tensorboardX import SummaryWriter


class TensorBoard:
    """
    Create a tensor board object.

    Attributes:
        writer: ``SummaryWriter`` of package ``tensorboardX``.
    """

    def __init__(self):
        self.writer = None

    def init(self, *writer_args):
        if self.writer is None:
            self.writer = SummaryWriter(*writer_args)
        else:
            raise RuntimeError("Writer has been initialized!")

    def is_inited(self) -> bool:
        """
        Returns: whether the board has been initialized with a writer.
        """
        return not self.writer is None


default_board = TensorBoard()
