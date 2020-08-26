def pytest_addoption(parser):
    parser.addoption("--gpu_device", action="store",
                     default="cuda:0",
                     help="Gpu device descriptor in pytorch")
