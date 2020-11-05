"""
Attributes:
    default_logger: The default global logger.

TODO: maybe add logging utilities for distributed scenario?
"""
import colorlog
from logging import INFO


class FakeLogger(object):
    def setLevel(self, level):
        """
        Sets the level for the given level.

        Args:
            self: (todo): write your description
            level: (str): write your description
        """
        pass

    def debug(self, msg, *args, **kwargs):
        """
        Log a debug message

        Args:
            self: (todo): write your description
            msg: (str): write your description
        """
        pass

    def info(self, msg, *args, **kwargs):
        """
        Log an info message

        Args:
            self: (todo): write your description
            msg: (str): write your description
        """
        pass

    def warning(self, msg, *args, **kwargs):
        """
        Log a warning.

        Args:
            self: (todo): write your description
            msg: (str): write your description
        """
        pass

    def warn(self, msg, *args, **kwargs):
        """
        Alias for warning.

        Args:
            self: (todo): write your description
            msg: (str): write your description
        """
        pass

    def error(self, msg, *args, **kwargs):
        """
        Log an error.

        Args:
            self: (todo): write your description
            msg: (str): write your description
        """
        pass

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """
        Exits a message with exception.

        Args:
            self: (todo): write your description
            msg: (str): write your description
            exc_info: (todo): write your description
        """
        pass

    def critical(self, msg, *args, **kwargs):
        """
        Log msg with severity severity.

        Args:
            self: (todo): write your description
            msg: (str): write your description
        """
        pass

    def log(self, level, msg, *args, **kwargs):
        """
        Log a message at level level level.

        Args:
            self: (todo): write your description
            level: (int): write your description
            msg: (str): write your description
        """
        pass


_default_handler = colorlog.StreamHandler()
_default_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s] <%(levelname)s>:%(name)s:%(message)s"))

default_logger = colorlog.getLogger("default_logger")
default_logger.addHandler(_default_handler)
default_logger.setLevel(INFO)
fake_logger = FakeLogger()

