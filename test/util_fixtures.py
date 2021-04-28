import torch as t
import pytest


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


__all__ = ["gpu", "device", "dtype", "mp_tmpdir"]
