import time


class Counter:
    def __init__(self):
        self._count = 0

    def count(self):
        self._count += 1

    def get(self):
        return self._count

    def reset(self):
        self._count = 0

    def __lt__(self, other):
        return self._count < other

    def __gt__(self, other):
        return self._count > other

    def __le__(self, other):
        return self._count <= other

    def __ge__(self, other):
        return self._count >= other

    def __eq__(self, other):
        return self._count == other

    def __repr__(self):
        return "%d" % self._count


class Switch:
    def __init__(self):
        self._on = False

    def flip(self):
        self._on = not self._on

    def get(self):
        return self._on

    def on(self):
        self._on = True

    def off(self):
        self._on = False


class Trigger:
    def __init__(self):
        self._on = False

    def flip(self):
        self._on = not self._on

    def get(self):
        on = self._on
        if self._on:
            self._on = False
        return on

    def on(self):
        self._on = True

    def off(self):
        self._on = False


class Timer:
    def __init__(self):
        self._last = time.time()

    def begin(self):
        self._last = time.time()

    def end(self):
        return time.time() - self._last


class Object:
    def __init__(self):
        self.data = {}

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return self.attr(item)

    def __setattr__(self, key, value):
        if key != "data" and key != "attr" and key != "call":
            self.attr(key, (value,))
        else:
            super(Object, self).__setattr__(key, value)

    def attr(self, item, change=None):
        pass