import os
import sys
import pytest
import pickle
import os.path as p
import subprocess as sp


class TestDDPPlugin:
    def test_all(self, tmpdir):
        test_save_path = str(p.join(tmpdir.make_numbered_dir(), "test.save"))
        env = os.environ.copy()
        env["TEST_SAVE_PATH"] = test_save_path
        process_0 = sp.Popen(
            [
                sys.executable,
                p.join(p.dirname(p.abspath(__file__)), "_pl_plugin_runner.py"),
                "ddp",
            ],
            env=env,
        )
        process_0.wait(timeout=20)
        if process_0.poll():
            process_0.kill()
            pytest.fail("Timeout on waiting for the DDPPlugin script to end.")
        elif process_0.returncode != 0:
            pytest.fail(f"DDPPlugin script returned code {process_0.returncode}.")
        else:
            with open(test_save_path, "rb") as f:
                flags = pickle.load(f)
                assert flags == [True], f"Not properly_inited, flags are: {flags}"


class TestDDPSpawnPlugin:
    def test_all(self, tmpdir):
        test_save_path = str(p.join(tmpdir.make_numbered_dir(), "test.save"))
        env = os.environ.copy()
        env["TEST_SAVE_PATH"] = test_save_path
        process_0 = sp.Popen(
            [
                sys.executable,
                p.join(p.dirname(p.abspath(__file__)), "_pl_plugin_runner.py"),
                "ddp_spawn",
            ],
            env=env,
        )
        process_0.wait(timeout=20)
        if process_0.poll():
            process_0.kill()
            pytest.fail("Timeout on waiting for the DDPSpawnPlugin script to end.")
        elif process_0.returncode != 0:
            pytest.fail(f"DDPSpawnPlugin script returned code {process_0.returncode}.")
        else:
            with open(test_save_path, "rb") as f:
                flags = pickle.load(f)
                assert flags == [True], f"Not properly_inited, flags are: {flags}"
