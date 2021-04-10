from datetime import datetime, timedelta
from typing import Union, Iterable
from os.path import join

import os
import shutil

from machin.utils.logging import default_logger
from machin.utils.prepare import prep_clear_dirs, prep_create_dirs


class SaveEnv:
    def __init__(
        self,
        env_root: str,
        restart_from_trial: Union[str, None] = None,
        time_format="%Y_%m_%d_%H_%M_%S",
        create_sub_dirs=True,
    ):
        """
        Create the default environment for saving. creates something like::

            <your environment root>
            ├── config
            ├── log
            │   ├── images
            │   └── train_log
            └── model

        Args:
            env_root: root directory for all trials of the environment.
            restart_from_trial: instead of creating a new save environment
                for a new trial, use a existing save environment of an older
                trial, old trial name should be in format ``time_format``
            time_format: Time formatter, setting it to an empty string will cause
                the save environment to use ``env_root`` directly instead of using
                sub directories with a datetime name.
            create_dirs: Whether to create directories.
        """
        self.env_root = env_root
        self.time_format = time_format

        if restart_from_trial is None:
            self.env_create_time = datetime.now()
        else:
            self.env_create_time = datetime.strptime(
                restart_from_trial, self.time_format
            )
        self._check_dirs()
        if create_sub_dirs:
            self.create_sub_dirs()

    def create_dirs(self, dirs: Iterable[str]):
        """
        Create additional directories in root.

        Args:
            dirs: Directories.
        """
        prep_create_dirs(
            [
                join(self.env_root, self.env_create_time.strftime(self.time_format), d)
                for d in dirs
            ]
        )

    def get_trial_root(self):
        # pylint: disable=missing-docstring
        return join(self.env_root, self.env_create_time.strftime(self.time_format))

    def get_trial_config_dir(self):
        # pylint: disable=missing-docstring
        return join(
            self.env_root, self.env_create_time.strftime(self.time_format), "config"
        )

    def get_trial_model_dir(self):
        # pylint: disable=missing-docstring
        return join(
            self.env_root, self.env_create_time.strftime(self.time_format), "model"
        )

    def get_trial_image_dir(self):
        # pylint: disable=missing-docstring
        return join(
            self.env_root,
            self.env_create_time.strftime(self.time_format),
            "log",
            "images",
        )

    def get_trial_train_log_dir(self):
        # pylint: disable=missing-docstring
        return join(
            self.env_root,
            self.env_create_time.strftime(self.time_format),
            "log",
            "train_log",
        )

    def get_trial_time(self):
        # pylint: disable=missing-docstring
        return self.env_create_time

    def clear_trial_config_dir(self):
        # pylint: disable=missing-docstring
        prep_clear_dirs(
            [
                join(
                    self.env_root,
                    self.env_create_time.strftime(self.time_format),
                    "config",
                )
            ]
        )

    def clear_trial_model_dir(self):
        # pylint: disable=missing-docstring
        prep_clear_dirs(
            [
                join(
                    self.env_root,
                    self.env_create_time.strftime(self.time_format),
                    "model",
                )
            ]
        )

    def clear_trial_image_dir(self):
        # pylint: disable=missing-docstring
        prep_clear_dirs(
            [
                join(
                    self.env_root,
                    self.env_create_time.strftime(self.time_format),
                    "log",
                    "images",
                )
            ]
        )

    def clear_trial_train_log_dir(self):
        # pylint: disable=missing-docstring
        prep_clear_dirs(
            [
                join(
                    self.env_root,
                    self.env_create_time.strftime(self.time_format),
                    "log",
                    "train_log",
                )
            ]
        )

    def remove_trials_older_than(
        self,
        diff_day: int = 0,
        diff_hour: int = 1,
        diff_minute: int = 0,
        diff_second: int = 0,
    ):
        """
        By default this function removes all trials started one hour earlier
        than current time.

        Args:
            diff_day: Difference in days.
            diff_hour: Difference in hours.
            diff_minute: Difference in minutes.
            diff_second: Difference in seconds.
        """
        trial_list = [f for f in os.listdir(self.env_root)]
        current_time = datetime.now()
        diff_threshold = timedelta(
            days=diff_day, hours=diff_hour, minutes=diff_minute, seconds=diff_second
        )
        for file in trial_list:
            try:
                time = datetime.strptime(file, self.time_format)
            except ValueError:
                # not a trial
                pass
            else:
                diff_time = current_time - time
                if diff_time > diff_threshold:
                    rm_path = join(self.env_root, file)
                    default_logger.info(f"Removing trial directory: {rm_path}")
                    shutil.rmtree(rm_path)

    def create_sub_dirs(self):
        root_dir = join(self.env_root, self.env_create_time.strftime(self.time_format))
        prep_create_dirs(
            (
                join(root_dir, "model"),
                join(root_dir, "config"),
                join(root_dir, "log", "images"),
                join(root_dir, "log", "train_log"),
            )
        )

    def _check_dirs(self):
        """
        Overload this function in your environment class to check directory
        mapping

        Raises:
            RuntimeError if directory mapping is invalid.
        """
        pass
