import os
import weakref
import itertools
from threading import Lock
from machin.utils.logging import default_logger

_finalizer_lock = Lock()
_finalizer_registry = {}
_finalizer_counter = itertools.count()


class Finalize(object):
    """
    Class which supports object finalization using weakrefs.
    Adapted from python 3.7.3 multiprocessing.util.
    """

    def __init__(self, obj, callback, args=(), kwargs=None, exitpriority=None):
        if (exitpriority is not None) and not isinstance(exitpriority, int):
            raise TypeError(
                "Exitpriority ({0!r}) must be None or int, not {1!s}".format(
                    exitpriority, type(exitpriority)
                )
            )

        if obj is not None:
            # weakref is just used to track the object in __repr__
            self._weakref = weakref.ref(obj, self)
        elif exitpriority is None:
            raise ValueError("Without object, exitpriority cannot be None")

        self._callback = callback
        self._args = args
        self._kwargs = kwargs or {}
        with _finalizer_lock:
            self._key = (exitpriority, next(_finalizer_counter))
        self._pid = os.getpid()

        _finalizer_registry[self._key] = self

    def __call__(
        self,
        # Need to bind these locally because the globals could have
        # been cleared at shutdown
        finalizer_registry=None,
        debug_logger=None,
        getpid=None,
    ):
        """
        Run the callback unless it has already been called or cancelled
        """
        finalizer_registry = finalizer_registry or _finalizer_registry
        debug_logger = debug_logger or default_logger
        getpid = getpid or os.getpid

        try:
            del finalizer_registry[self._key]
        except KeyError:
            debug_logger.debug("finalizer no longer registered")
        else:
            if self._pid != getpid():
                debug_logger.debug("finalizer ignored because different process")
                res = None
            else:
                debug_logger.debug(
                    f"finalizer calling {self._callback} "
                    f"with args {self._args} "
                    f"and kwargs {self._kwargs}"
                )
                res = self._callback(*self._args, **self._kwargs)
            self._weakref = (
                self._callback
            ) = self._args = self._kwargs = self._key = None
            return res

    def cancel(self):
        """
        Cancel finalization of the object
        """
        try:
            del _finalizer_registry[self._key]
        except KeyError:
            pass
        else:
            self._weakref = (
                self._callback
            ) = self._args = self._kwargs = self._key = None

    def still_active(self):
        """
        Return whether this finalizer is still waiting to invoke callback
        """
        return self._key in _finalizer_registry

    def __repr__(self):
        try:
            obj = self._weakref()
        except (AttributeError, TypeError):
            obj = None

        if obj is None:
            return f"<{self.__class__.__name__} object, dead>"

        x = (
            f"<{self.__class__.__name__} object, "
            f"callback={getattr(self._callback, '__name__', self._callback)}"
        )
        if self._args:
            x += ", args=" + str(self._args)
        if self._kwargs:
            x += ", kwargs=" + str(self._kwargs)
        if self._key[0] is not None:
            x += ", exitprority=" + str(self._key[0])
        return x + ">"
