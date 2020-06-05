from torchviz import make_dot
"""
The visualization module, currently it only contains the pytorch
flow graph visualization, more visualizations for cnn, resnet, lstm & rnn,
attention layers will be added in the future, if there is any feature request.
"""


def visualize_graph(final_tensor, exit_after_vis=True):
    """
    Visualize a pytorch flow graph

    Args:
        final_tensor: The last output tensor of the flow graph
        exit_after_vis: Whether to exit the whole program
            after visualization.
    """
    g = make_dot(final_tensor)
    g.view(quiet_view=True, quiet=True)
    if exit_after_vis:
        exit(0)
