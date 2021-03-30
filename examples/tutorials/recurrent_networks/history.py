import torch as t


class History:
    def __init__(self, history_depth, state_shape):
        self.history = [t.zeros(state_shape) for _ in range(history_depth)]
        self.state_shape = state_shape

    def append(self, state):
        assert (
            t.is_tensor(state)
            and state.dtype == t.float32
            and tuple(state.shape) == self.state_shape
        )
        self.history.append(state)
        self.history.pop(0)
        return self

    def get(self):
        # size: (1, history_depth, ...)
        return t.cat(self.history, dim=0).unsqueeze(0)
