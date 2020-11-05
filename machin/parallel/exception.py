import traceback


class RemoteTraceback(Exception):  # pragma: no cover
    """
    Remote traceback, rebuilt by ``_rebuild_exc`` from pickled original
    traceback ExceptionWithTraceback, should be thrown on the master
    side.
    """
    def __init__(self, tb):
        """
        Initialize a tb

        Args:
            self: (todo): write your description
            tb: (int): write your description
        """
        self.tb = tb

    def __str__(self):
        """
        Returns the string : class

        Args:
            self: (todo): write your description
        """
        return self.tb


def _rebuild_exc(exc, tb):
    """
    Rebuilds exceptions.

    Args:
        exc: (todo): write your description
        tb: (todo): write your description
    """
    exc.__cause__ = RemoteTraceback(tb)
    return exc


class ExceptionWithTraceback:  # pragma: no cover
    def __init__(self, exc: Exception, tb: str = None):
        """
        This exception class is used by slave processes to capture
        exceptions thrown during execution and send back throw queues
        to their master.

        Args:
            exc: Your exception.
            tb: An optional traceback, by default is is set to
                ``exc.__traceback__``
        """
        if tb is None:
            tb = exc.__traceback__
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        self.tb = '\n"""\n%s"""' % tb

    def __reduce__(self):
        """
        Reduce a reduce.

        Args:
            self: (todo): write your description
        """
        # Used by pickler
        return _rebuild_exc, (self.exc, self.tb)


