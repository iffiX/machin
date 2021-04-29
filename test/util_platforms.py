import sys
import pytest

linux_only = pytest.mark.skipif("not sys.platform.startswith('linux')")
windows_only = pytest.mark.skipif("not sys.platform.startswith('win')")
macos_only = pytest.mark.skipif("not sys.platform.startswith('darwin')")


def linux_only_forall():
    if not sys.platform.startswith("linux"):
        pytest.skip("Requires Linux platform", allow_module_level=True)


def windows_only_forall():
    if not sys.platform.startswith("win"):
        pytest.skip("Requires Windows platform", allow_module_level=True)


def macos_only_forall():
    if not sys.platform.startswith("darwin"):
        pytest.skip("Requires MacOS platform", allow_module_level=True)
