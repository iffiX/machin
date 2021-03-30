"""
Attributes:
    default_logger: The default global logger.

TODO: maybe add logging utilities for distributed scenario?
"""
import colorlog
from logging import INFO


class FakeLogger:
    def setLevel(self, level):
        pass

    def debug(self, msg, *args, **kwargs):
        pass

    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def warn(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

    def exception(self, msg, *args, exc_info=True, **kwargs):
        pass

    def critical(self, msg, *args, **kwargs):
        pass

    def log(self, level, msg, *args, **kwargs):
        pass


_default_handler = colorlog.StreamHandler()
_default_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] <%(levelname)s>:%(name)s:%(message)s"
    )
)

default_logger = colorlog.getLogger("default_logger")
default_logger.addHandler(_default_handler)
default_logger.setLevel(INFO)
fake_logger = FakeLogger()
