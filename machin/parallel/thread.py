import traceback
import threading


class ThreadException(Exception):
    pass


class Thread(threading.Thread):
    """
    Enhanced thread with exception tracing.
    """
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, cleaner=None, *, daemon=None):
        """
        Initialize the daemon.

        Args:
            self: (todo): write your description
            group: (todo): write your description
            target: (todo): write your description
            name: (str): write your description
            cleaner: (todo): write your description
            daemon: (todo): write your description
        """
        threading.Thread.__init__(self,
                                  group=group, target=target,
                                  name=name, args=args, kwargs=kwargs,
                                  daemon=daemon)
        self._cleaner = cleaner
        self._exception_str = ""
        self._has_exception = False

    @property
    def exception(self):
        """
        Returns the exception object for the given exception.

        Args:
            self: (todo): write your description
        """
        if not self._has_exception:
            return None
        exc = ThreadException(self._exception_str)
        return exc

    def watch(self):
        """
        Watch the event.

        Args:
            self: (todo): write your description
        """
        if self._has_exception:
            raise self.exception

    @staticmethod
    def format_exceptions(exceptions):
        """
        Formats the traceback exceptions.

        Args:
            exceptions: (todo): write your description
        """
        all_tb = ""
        for exc, i in zip(exceptions, range(len(exceptions))):
            tb = exc.__traceback__
            tb = traceback.format_exception(type(exc), exc, tb)
            tb = "".join(tb)
            all_tb = all_tb + "\nException {}:\n{}".format(i, tb)
        return all_tb

    def run(self):
        """
        Runs the exception.

        Args:
            self: (todo): write your description
        """
        exc = []
        try:
            super(Thread, self).run()
        except BaseException as e:
            exc.append(e)
        finally:
            if self._cleaner is not None:
                try:
                    self._cleaner()
                except BaseException as e:
                    exc.append(e)
            if exc:
                self._exception_str = self.format_exceptions(exc)
                self._has_exception = True
