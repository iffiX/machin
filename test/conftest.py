def pytest_addoption(parser):
    """
    Add a pytest option. py file.

    Args:
        parser: (todo): write your description
    """
    parser.addoption("--gpu_device", action="store",
                     default="cuda:0",
                     help="Gpu device descriptor in pytorch")
