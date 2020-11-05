from machin.utils.tensor_board import default_board
import pytest


class TestTensorBoard(object):
    def test_tensor_board(self):
        """
        Test if the board.

        Args:
            self: (todo): write your description
        """
        default_board.init()
        with pytest.raises(RuntimeError, match="has been initialized"):
            default_board.init()
        assert default_board.is_inited()
