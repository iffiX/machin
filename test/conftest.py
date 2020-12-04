def pytest_addoption(parser):
    parser.addoption("--gpu_device", action="store",
                     default=None,
                     help="Gpu device descriptor in pytorch")
