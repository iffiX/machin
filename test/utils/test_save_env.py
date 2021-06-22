from os.path import join
from machin.utils.save_env import SaveEnv

import os
import time


class TestSaveEnv:
    def test_all(self, tmpdir):
        tmp_dir = str(tmpdir.make_numbered_dir())
        save_env = SaveEnv(env_root=tmp_dir)
        save_env = SaveEnv(
            env_root=tmp_dir,
            restart_from_trial=os.path.basename(save_env.get_trial_root()),
        )

        # check directories
        t_root = save_env.get_trial_root()
        assert (
            os.path.exists(join(t_root, "model"))
            and os.path.isdir(join(t_root, "model"))
            and os.path.exists(join(t_root, "config"))
            and os.path.isdir(join(t_root, "config"))
            and os.path.exists(join(t_root, "log", "images"))
            and os.path.isdir(join(t_root, "log", "images"))
            and os.path.exists(join(t_root, "log", "train_log"))
            and os.path.isdir(join(t_root, "log", "train_log"))
        )

        save_env.create_dirs(["some_custom_dir"])
        assert os.path.exists(join(t_root, "some_custom_dir")) and os.path.isdir(
            join(t_root, "some_custom_dir")
        )

        save_env.get_trial_time()
        assert save_env.get_trial_config_dir() == join(t_root, "config")
        assert save_env.get_trial_model_dir() == join(t_root, "model")
        assert save_env.get_trial_image_dir() == join(t_root, "log", "images")
        assert save_env.get_trial_train_log_dir() == join(t_root, "log", "train_log")

        with open(join(t_root, "config", "conf.json"), "w") as _:
            pass
        with open(join(t_root, "model", "model.pt"), "w") as _:
            pass
        with open(join(t_root, "log", "images", "image.png"), "w") as _:
            pass
        with open(join(t_root, "log", "train_log", "log.txt"), "w") as _:
            pass
        save_env.clear_trial_config_dir()
        assert not os.path.exists(join(t_root, "config", "conf.json"))
        save_env.clear_trial_model_dir()
        assert not os.path.exists(join(t_root, "model", "model.pt"))
        save_env.clear_trial_image_dir()
        assert not os.path.exists(join(t_root, "log", "images", "image.png"))
        save_env.clear_trial_train_log_dir()
        assert not os.path.exists(join(t_root, "log", "train_log", "log.txt"))

        time.sleep(2)
        os.mkdir(join(tmp_dir, "some_dir_not_trial"))
        save_env2 = SaveEnv(env_root=tmp_dir)
        save_env2.remove_trials_older_than(0, 0, 0, 1)
        assert not os.path.exists(t_root)
