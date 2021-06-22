from machin.utils.tensor_board import default_board

import pytest


class TestTensorBoard:
    def test_tensor_board(self):
        default_board.init()
        with pytest.raises(RuntimeError, match="has been initialized"):
            default_board.init()
        assert default_board.is_inited()
