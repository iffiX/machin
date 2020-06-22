from machin.utils.visualize import visualize_graph
import mock
import torch as t


def mock_exit(code):
    pass


def test_visualize_graph():
    tensor = t.ones([2, 2])
    tensor = tensor * t.ones([2, 2])
    with mock.patch('machin.utils.visualize.exit', mock_exit):
        visualize_graph(tensor)
