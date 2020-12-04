import torch as t
import pytest


@pytest.fixture()
def gpu(pytestconfig):
    return pytestconfig.getoption("gpu_device") or "cpu"


@pytest.fixture(params=["cpu", "gpu"])
def device(pytestconfig, request):
    if request.param == "cpu":
        return "cpu"
    return pytestconfig.getoption("gpu_device", skip=True)


@pytest.fixture(params=["float32", "float64"])
def dtype(pytestconfig, request):
    if request.param == "float32":
        return t.float32
    return t.float64


__all__ = ["gpu", "device", "dtype"]
