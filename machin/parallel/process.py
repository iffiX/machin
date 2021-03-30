from multiprocessing import Pipe
from multiprocessing.process import BaseProcess
from multiprocessing.context import _default_context
import traceback


class ProcessException(Exception):
    pass


class Process(BaseProcess):
    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs={},
        cleaner=None,
        ctx=_default_context,
        *,
        daemon=None,
    ):
        self._exc_pipe = Pipe()
        super().__init__(
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs=kwargs,
            daemon=daemon,
        )
        self._cleaner = cleaner
        self._ctx = ctx
        self._start_method = ctx.Process._start_method

    @staticmethod
    def _Popen(process_obj):
        assert isinstance(process_obj, Process)
        return process_obj._ctx.Process._Popen(process_obj)

    @property
    def exception(self):
        if not self._exc_pipe[0].poll(timeout=1e-4):
            return None
        exc = ProcessException(self._exc_pipe[0].recv())
        return exc

    def watch(self):
        if self._exc_pipe[0].poll(timeout=1e-4):
            self.join()
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
                self._exc_pipe[1].send(self.format_exceptions(exc))
