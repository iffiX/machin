import torch as t
import pytest
from test.data.archive import Archive
from test.data.all import generate_all, get_all


@pytest.fixture()
def gpu(pytestconfig):
    dev = pytestconfig.getoption("gpu_device")
    if dev is not None and dev.startswith("cuda"):
        return dev
    pytest.skip(f"Requiring GPU but provided `gpu_device` is {dev}")


@pytest.fixture(params=["cpu", "gpu"])
def device(pytestconfig, request):
    if request.param == "cpu":
        return "cpu"
    else:
        dev = pytestconfig.getoption("gpu_device")
        if dev is not None and dev.startswith("cuda"):
            return dev
        pytest.skip(f"Requiring GPU but provided `gpu_device` is {dev}")


@pytest.fixture(params=["float32", "float64"])
def dtype(pytestconfig, request):
    if request.param == "float32":
        return t.float32
    return t.float64


@pytest.fixture()
def mp_tmpdir(tmpdir):
    """
    For multiprocessing, sharing the same tmpdir across all processes
    """
    return tmpdir.make_numbered_dir()


@pytest.fixture(scope="session")
def archives():
    # prepare all test data archives
    generate_all()
    return get_all()


__all__ = ["gpu", "device", "dtype", "mp_tmpdir", "archives"]
