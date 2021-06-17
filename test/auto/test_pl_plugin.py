from test.util_platforms import linux_only_forall
import os
import sys
import pytest
import pickle
import os.path as p
import subprocess as sp


linux_only_forall()


class TestDDPPlugin:
    def test_all(self, tmpdir):
        test_save_path = str(p.join(tmpdir.make_numbered_dir(), "test.save"))
        env = os.environ.copy()
        env["TEST_SAVE_PATH"] = test_save_path
        processes = [
            sp.Popen(
                [
                    sys.executable,
                    p.join(p.dirname(p.abspath(__file__)), "_pl_plugin_runner.py"),
                    "ddp",
                    str(i),
                ],
                env=env,
            )
            for i in range(3)
        ]
        try:
            for process in processes:
                process.wait(timeout=20)
        except sp.TimeoutExpired:
            pytest.fail("Timeout on waiting for the DDPPlugin script to end.")

        with open(test_save_path, "rb") as f:
            flags = pickle.load(f)
            assert flags == [True], f"Not properly_inited, flags are: {flags}"


class TestDDPSpawnPlugin:
    def test_all(self, tmpdir):
        test_save_path = str(p.join(tmpdir.make_numbered_dir(), "test.save"))
        env = os.environ.copy()
        env["TEST_SAVE_PATH"] = test_save_path
        processes = [
            sp.Popen(
                [
                    sys.executable,
                    p.join(p.dirname(p.abspath(__file__)), "_pl_plugin_runner.py"),
                    "ddp_spawn",
                    str(i),
                ],
                env=env,
            )
            for i in range(3)
        ]
        try:
            for process in processes:
                process.wait(timeout=20)
        except sp.TimeoutExpired:
            pytest.fail("Timeout on waiting for the DDPSpawnPlugin script to end.")

        with open(test_save_path, "rb") as f:
            flags = pickle.load(f)
            assert flags == [True], f"Not properly_inited, flags are: {flags}"
