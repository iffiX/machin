import traceback
import threading


class ThreadException(Exception):
    pass


class Thread(threading.Thread):
    """
    Enhanced thread with exception tracing.
    """

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs={},
        cleaner=None,
        *,
        daemon=None,
    ):
        threading.Thread.__init__(
            self,
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs=kwargs,
            daemon=daemon,
        )
        self._cleaner = cleaner
        self._exception_str = ""
        self._has_exception = False

    @property
    def exception(self):
        if not self._has_exception:
            return None
        exc = ThreadException(self._exception_str)
        return exc

    def watch(self):
        if self._has_exception:
            raise self.exception

    @staticmethod
    def format_exceptions(exceptions):
        all_tb = ""
        for exc, i in zip(exceptions, range(len(exceptions))):
            tb = exc.__traceback__
            tb = traceback.format_exception(type(exc), exc, tb)
            tb = "".join(tb)
            all_tb = all_tb + f"\nException {i}:\n{tb}"
        return all_tb

    def run(self):
        exc = []
        try:
            super().run()
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
