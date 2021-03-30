import time


class Counter:
    def __init__(self, start=0, step=1):
        self._count = start
        self._start = start
        self._step = step

    def count(self):
        """
        Move counter forward by ``step``
        """
        self._count += self._step

    def get(self):
        """
        Get the internal number of counter.
        """
        return self._count

    def reset(self):
        """
        Reset the counter.
        """
        self._count = self._start

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
    def __init__(self, state: bool = False):
        """
        Args:
            state: Internal state, ``True`` for on, ``False`` for off.
        """
        self._on = state

    def flip(self):
        """
        Inverse the internal state.
        """
        self._on = not self._on

    def get(self) -> bool:
        """
        Returns:
            state of switch.
        """
        return self._on

    def on(self):
        """
        Set to on.
        """
        self._on = True

    def off(self):
        """
        Set to off.
        """
        self._on = False


class Trigger(Switch):
    def get(self):
        """
        Get the state of trigger, will also set trigger to off.

        Returns:
            state of trigger.
        """
        on = self._on
        if self._on:
            self._on = False
        return on


class Timer:
    def __init__(self):
        self._last = time.time()

    def begin(self):
        """
        Begin timing.
        """
        self._last = time.time()

    def end(self):
        """
        Returns:
            Curent time difference since last ``begin()``
        """
        return time.time() - self._last


class Object:
    """
    An generic object class, which stores a dictionary internally, and you
    can access and set its keys by accessing and seting attributes of the
    object.

    Attributes:
        data: Internal dictionary.
    """

    def __init__(self, data=None, const_attrs=None):
        if data is None:
            data = {}
        if const_attrs is None:
            const_attrs = set()
        super().__setattr__("const_attrs", const_attrs)
        super().__setattr__("data", data)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        """
        the implementation of ``Object.__call__``, override it to
        customize call behavior.
        """
        pass

    def __getattr__(self, item):
        # This function will be called if python cannot find an attribute
        # Note: in order to make Object picklable, we must raise AttributeError
        # when looking for special methods, such that when pickler is looking
        # up a non-existing __getstate__ function etc, this class will
        # not return a None value because self.attr(item) will return None.
        if isinstance(item, str) and item[:2] == item[-2:] == "__":
            # skip non-existing special method lookups
            raise AttributeError(f"Failed to find attribute: {item}")
        return self.attr(item)

    def __getitem__(self, item):
        return self.attr(item)

    def __setattr__(self, key, value):
        if (
            key != "data"
            and key != "attr"
            and key != "call"
            and key not in self.__dir__()
        ):
            if key in self.const_attrs:
                raise RuntimeError(f"{key} is const.")
            self.attr(key, value, change=True)
        elif key == "call":
            super().__setattr__(key, value)
        elif key == "data":
            if isinstance(value, dict):
                super().__setattr__(key, value)
            else:
                raise ValueError("The data attribute must be a dictionary.")
        else:
            raise RuntimeError(
                f"You should not set the {key} property of an "
                "Object. You can only set non-const keys "
                "in data and .data and .call attributes."
            )

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def attr(self, item, value=None, change=False):
        if change:
            self.data[item] = value
        return self.data.get(item, None)
