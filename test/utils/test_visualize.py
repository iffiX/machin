from machin.utils.visualize import visualize_graph
import mock
import torch as t


def mock_exit(_exit_code):
    """
    Makes a exit_code

    Args:
        _exit_code: (str): write your description
    """
    pass


def test_visualize_graph():
    """
    Visualize a graph.

    Args:
    """
    tensor = t.ones([2, 2])
    tensor = tensor * t.ones([2, 2])
    with mock.patch('machin.utils.visualize.exit', mock_exit):
        visualize_graph(tensor)
