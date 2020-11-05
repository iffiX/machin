import torch as t


class History:
    def __init__(self, history_depth, state_shape):
        """
        Initialize the state.

        Args:
            self: (todo): write your description
            history_depth: (int): write your description
            state_shape: (todo): write your description
        """
        self.history = [t.zeros(state_shape) for _ in range(history_depth)]
        self.state_shape = state_shape

    def append(self, state):
        """
        Append a new tensor to the buffer.

        Args:
            self: (todo): write your description
            state: (todo): write your description
        """
        assert (t.is_tensor(state) and
                state.dtype == t.float32 and
                tuple(state.shape) == self.state_shape)
        self.history.append(state)
        self.history.pop(0)
        return self

    def get(self):
        """
        Return the history of this node.

        Args:
            self: (todo): write your description
        """
        # size: (1, history_depth, ...)
        return t.cat(self.history, dim=0).unsqueeze(0)
