from torchviz import make_dot


def visualize_graph(final_tensor):
    g = make_dot(final_tensor)
    g.view()
