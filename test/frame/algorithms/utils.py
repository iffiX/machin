from gym.wrappers.time_limit import TimeLimit


def unwrap_time_limit(env):
    # some environment comes with a time limit, we must remove it
    if isinstance(env, TimeLimit):
        return env.unwrapped
    else:
        return env


class Smooth(object):
    def __init__(self, init_value=0):
        self._value = init_value

    def update(self, new_value, update_rate=0.2):
        self._value = self._value * (1 - update_rate) + new_value * update_rate
        return self._value

    @property
    def value(self):
        return self._value