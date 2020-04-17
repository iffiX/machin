"""A logging formatter for colored output."""

from __future__ import absolute_import
from logging import *
from utils.colorlog.colorlog import (
    escape_codes, default_log_colors,
    ColoredFormatter, LevelFormatter, TTYColoredFormatter)

from utils.colorlog.logging import (
    basicConfig, root, getLogger, log,
    debug, info, warning, error, exception, critical, StreamHandler)

__all__ = ('ColoredFormatter', 'default_log_colors', 'escape_codes',
           'basicConfig', 'root', 'getLogger', 'debug', 'info', 'warning',
           'error', 'exception', 'critical', 'log', 'exception',
           'StreamHandler', 'LevelFormatter', 'TTYColoredFormatter')
