from torchviz import make_dot


def visualize_graph(final_tensor, exit_after_vis=True):
    g = make_dot(final_tensor)
    g.view()
    if exit_after_vis:
        exit(0)
