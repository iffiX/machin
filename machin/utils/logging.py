"""
Attributes:
    default_logger: The default global logger.

TODO: maybe add logging utilities for distributed scenario?
"""
import colorlog
from logging import INFO

_default_handler = colorlog.StreamHandler()
_default_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s] <%(levelname)s>:%(name)s:%(message)s"))

default_logger = colorlog.getLogger("default_logger")
default_logger.addHandler(_default_handler)
default_logger.setLevel(INFO)
