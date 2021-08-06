def pytest_addoption(parser):
    parser.addoption(
        "--gpu_device",
        action="store",
        default=None,
        help="GPU device descriptor in pytorch",
    )
    parser.addoption(
        "--multiprocess_method",
        default="forkserver",
        help="spawn or forkserver, default is forkserver",
    )
