from multiprocessing import Pipe
from multiprocessing.process import BaseProcess
from multiprocessing.context import _default_context
import traceback


class ProcessException(Exception):
    pass


class Process(BaseProcess):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={},
                 cleaner=None, ctx=_default_context, *, daemon=None):
        """
        Initialize a new process.

        Args:
            self: (todo): write your description
            group: (todo): write your description
            target: (todo): write your description
            name: (str): write your description
            cleaner: (todo): write your description
            ctx: (str): write your description
            _default_context: (str): write your description
            daemon: (todo): write your description
        """
        self._exc_pipe = Pipe()
        super(Process, self).__init__(group=group, target=target,
                                      name=name, args=args, kwargs=kwargs,
                                      daemon=daemon)
        self._cleaner = cleaner
        self._ctx = ctx
        self._start_method = ctx.Process._start_method

    @staticmethod
    def _Popen(process_obj):
        """
        Executes a process.

        Args:
            process_obj: (todo): write your description
        """
        assert isinstance(process_obj, Process)
        return process_obj._ctx.Process._Popen(process_obj)

    @property
    def exception(self):
        """
        Returns the exception raised by the exception.

        Args:
            self: (todo): write your description
        """
        if not self._exc_pipe[0].poll(timeout=1e-4):
            return None
        exc = ProcessException(self._exc_pipe[0].recv())
        return exc

    def watch(self):
        """
        Watch for the watch.

        Args:
            self: (todo): write your description
        """
        if self._exc_pipe[0].poll(timeout=1e-4):
            self.join()
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
        Run the given exception.

        Args:
            self: (todo): write your description
        """
        exc = []
        try:
            super(Process, self).run()
        except BaseException as e:
            exc.append(e)
        finally:
            if self._cleaner is not None:
                try:
                    self._cleaner()
                except BaseException as e:
                    exc.append(e)
            if exc:
                self._exc_pipe[1].send(self.format_exceptions(exc))
