from gym.wrappers.time_limit import TimeLimit


def unwrap_time_limit(env):
    # some environment comes with a time limit, we must remove it
    if isinstance(env, TimeLimit):
        return env.unwrapped
    else:
        return env


class Smooth:
    def __init__(self):
        self._value = None

    def update(self, new_value, update_rate=0.2):
        if self._value is None:
            self._value = new_value
        else:
            self._value = self._value * (1 - update_rate) + new_value * update_rate
        return self._value

    @property
    def value(self):
        return self._value
