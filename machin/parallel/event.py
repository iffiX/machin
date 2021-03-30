from threading import Event, Lock
import time

_lock = Lock()


def init_or_register(event, parent_event):
    if hasattr(event, "_magic_parent_events"):
        event._magic_parent_events.append(parent_event)
    else:
        org_set = event.set
        event._magic_parent_events = [parent_event]

        def new_set():
            nonlocal event, org_set
            for pe in event._magic_parent_events:
                with pe._cond:
                    pe._cond.notify_all()
            org_set()

        event.set = new_set


class MultiEvent(Event):
    # can only be used with Event from threading and not from multiprocessing
    def __init__(self, *events):
        global _lock
        super().__init__()

        with _lock:
            self.is_leaf = all([type(e) == Event for e in events])
            for event in events:
                if type(event) == Event:
                    init_or_register(event, self)
                elif isinstance(event, MultiEvent):
                    for le in event.get_leaf_events():
                        init_or_register(le, self)
                else:
                    raise ValueError(
                        f"Type {type(event)} is not a valid event type, "
                        "requires threading.Event or an instance of MultiEvent."
                    )
        self._events = events

    def set(self):
        pass

    def clear(self):
        pass

    def is_set(self):
        return False

    def get_leaf_events(self):
        for event in self._events:
            if type(event) == Event:
                yield event
            else:
                for event in event.get_leaf_events():
                    yield event

    def wait(self, timeout=None):
        begin = time.time()
        while True:
            with self._cond:
                self._cond.wait(timeout)
            if timeout is None or (
                timeout is not None and time.time() - begin >= timeout
            ):
                break
        return self.is_set()


class OrEvent(MultiEvent):
    def is_set(self):
        return any([event.is_set() for event in self._events])


class AndEvent(MultiEvent):
    def is_set(self):
        return all([event.is_set() for event in self._events])
