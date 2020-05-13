import os
from datetime import datetime, timedelta
from typing import Union

from utils.prep import prep_dirs_default, prep_clear_dirs, prep_create_dirs


class SaveEnv:
    def __init__(self,
                 env_root: str,
                 env_dirs: Union[dict, None]=None,
                 restart_use_trial: Union[str, None]=None):
        """
        Create the default environment for saving.

        Args:
            env_root: root directory for all trials of the environment.
            env_dirs: directory mapping for sub directories, such as log, model, etc.
            restart_use_trial: in format %Y_%m_%d_%H_%M_%S
        """
        self.env_root = env_root
        self.env_dirs = env_dirs

        if restart_use_trial is None:
            self.env_create_time = datetime.now()
        else:
            self.env_create_time = datetime.strptime(restart_use_trial,
                                                     "%Y_%m_%d_%H_%M_%S")
        self._check_dirs()
        self._prep_dirs()

    def create_dirs(self, dirs):
        prep_create_dirs(dirs)

    def get_trial_root(self):
        return os.path.join(self.env_root,
                            self.env_create_time.strftime("%Y_%m_%d_%H_%M_%S"))

    def get_trial_config_file(self):
        return os.path.join(self.env_root,
                            self.env_create_time.strftime("%Y_%m_%d_%H_%M_%S"),
                            "config/config.json")

    def get_trial_model_dir(self):
        return os.path.join(self.env_root,
                            self.env_create_time.strftime("%Y_%m_%d_%H_%M_%S"),
                            "model")

    def get_trial_image_dir(self):
        return os.path.join(self.env_root,
                            self.env_create_time.strftime("%Y_%m_%d_%H_%M_%S"),
                            "log/images")

    def get_trial_train_log_dir(self):
        return os.path.join(self.env_root,
                            self.env_create_time.strftime("%Y_%m_%d_%H_%M_%S"),
                            "log/train_log")

    def get_trial_time(self):
        return self.env_create_time

    def clear_trial_config_dir(self):
        prep_clear_dirs([
            os.path.join(self.env_root,
                         self.env_create_time.strftime("%Y_%m_%d_%H_%M_%S"),
                         "config")
        ])

    def clear_trial_model_dir(self):
        prep_clear_dirs([
            os.path.join(self.env_root,
                         self.env_create_time.strftime("%Y_%m_%d_%H_%M_%S"),
                         "model")
        ])

    def clear_trial_image_dir(self):
        prep_clear_dirs([
            os.path.join(self.env_root,
                         self.env_create_time.strftime("%Y_%m_%d_%H_%M_%S"),
                         "log/images")
        ])

    def clear_trial_train_log_dir(self):
        prep_clear_dirs([
            os.path.join(self.env_root,
                         self.env_create_time.strftime("%Y_%m_%d_%H_%M_%S"),
                         "log/train_log")
        ])

    def remove_trials_older_than(self,
                                 diff_day=0,
                                 diff_hour=1,
                                 diff_minute=0,
                                 diff_second=0):
        """
        By default this function removes all trials started one hour earlier than current time.
        """
        trial_list = [f for f in os.listdir(self.env_root)]
        current_time = datetime.now()
        diff_threshold = timedelta(days=diff_day, hours=diff_hour,
                                   minutes=diff_minute, seconds=diff_second)
        for file in trial_list:
            try:
                time = datetime.strptime(file, "%Y_%m_%d_%H_%M_%S")
            except Exception:
                # not a trial
                pass
            else:
                diff_time = current_time - time
                if diff_time > diff_threshold:
                    rm_path = os.path.join(self.env_root, file)
                    print("Removing trial directory: {}".format(rm_path))
                    os.remove(rm_path)

    def _prep_dirs(self):
        root_dir = os.path.join(self.env_root,
                                self.env_create_time.strftime("%Y_%m_%d_%H_%M_%S"))
        prep_dirs_default(root_dir)

    def _check_dirs(self):
        """
        Overload this function in your environment class to check directory mapping
        raise RuntimeError if directory mapping is invalid
        """
        pass