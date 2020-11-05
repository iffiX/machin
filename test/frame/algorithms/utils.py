from gym.wrappers.time_limit import TimeLimit


def unwrap_time_limit(env):
    """
    Wraps a function that can be used as a future.

    Args:
        env: (todo): write your description
    """
    # some environment comes with a time limit, we must remove it
    if isinstance(env, TimeLimit):
        return env.unwrapped
    else:
        return env


class Smooth(object):
    def __init__(self):
        """
        Initialize the internal state of the class.

        Args:
            self: (todo): write your description
        """
        self._value = None

    def update(self, new_value, update_rate=0.2):
        """
        Update the rate.

        Args:
            self: (todo): write your description
            new_value: (todo): write your description
            update_rate: (todo): write your description
        """
        if self._value is None:
            self._value = new_value
        else:
            self._value = (self._value * (1 - update_rate)
                           + new_value * update_rate)
        return self._value

    @property
    def value(self):
        """
        Returns the value of the field.

        Args:
            self: (todo): write your description
        """
        return self._value
