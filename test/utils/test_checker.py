from machin.utils.checker import (
    CheckError,
    check_shape,
    check_nan,
    check_model,
    mark_as_atom_module,
    mark_module_output,
    p_chk_nan,
    p_chk_range
)
from machin.utils.tensor_board import TensorBoard
import pytest
import torch as t
import torch.nn as nn


def test_check_shape():
    """
    R check that the tensors.

    Args:
    """
    with pytest.raises(CheckError, match="has invalid shape"):
        tensor = t.zeros([10, 10])
        check_shape(tensor, [5, 5])


def test_check_nan():
    """
    Check if tensor is nan.

    Args:
    """
    with pytest.raises(CheckError, match="contains nan"):
        tensor = t.full([10, 10], float('NaN'))
        check_nan(tensor)


class SubModule1(nn.Module):
    def __init__(self):
        """
        Initialize the module.

        Args:
            self: (todo): write your description
        """
        super(SubModule1, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 20)
        mark_as_atom_module(self)
        mark_module_output(self, ["output1_sub1"])

    def forward(self, x):
        """
        Forward forward forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.fc2(self.fc1(x)), None


class SubModule2(nn.Module):
    def __init__(self):
        """
        Initialize the super1.

        Args:
            self: (todo): write your description
        """
        super(SubModule2, self).__init__()
        self.fc1 = nn.Linear(20, 20)

    def forward(self, x):
        """
        Return the forward forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.fc1(x)


class CheckedModel(nn.Module):
    def __init__(self):
        """
        Initializes sub1 and sub2

        Args:
            self: (todo): write your description
        """
        super(CheckedModel, self).__init__()
        self.sub1 = SubModule1()
        self.sub2 = SubModule2()

    def forward(self, x):
        """
        Return the sub1 of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.sub2(self.sub1(x)[0])


param_checked = False


def param_check_hook(*_):
    """
    Decorator that the global hook.

    Args:
        _: (todo): write your description
    """
    global param_checked
    param_checked = True


def test_check_model():
    """
    Check that all checks.

    Args:
    """
    global param_checked
    board = TensorBoard()
    board.init()
    model = CheckedModel()
    cancel = check_model(board.writer,
                         model,
                         param_check_interval=1,
                         param_check_hooks=(param_check_hook,
                                            p_chk_nan,
                                            p_chk_range),
                         name="checked_model")
    output = model(t.ones([1, 5]))
    output.sum().backward()
    cancel()
    assert param_checked
