import numpy as np
from pyglet.window import key

# individual agent policy
class Policy(object):
    def __init__(self):
        """
        Initialize the object

        Args:
            self: (todo): write your description
        """
        pass
    def action(self, obs):
        """
        Returns the action.

        Args:
            self: (todo): write your description
            obs: (todo): write your description
        """
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        """
        Initialize the window.

        Args:
            self: (todo): write your description
            env: (todo): write your description
            agent_index: (str): write your description
        """
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        """
        Return an action.

        Args:
            self: (todo): write your description
            obs: (todo): write your description
        """
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 1.0
            if self.move[1]: u[2] += 1.0
            if self.move[3]: u[3] += 1.0
            if self.move[2]: u[4] += 1.0
            if True not in self.move:
                u[0] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        """
        Perform a key press.

        Args:
            self: (todo): write your description
            k: (todo): write your description
            mod: (todo): write your description
        """
        if k==key.LEFT:  self.move[0] = True
        if k==key.RIGHT: self.move[1] = True
        if k==key.UP:    self.move[2] = True
        if k==key.DOWN:  self.move[3] = True
    def key_release(self, k, mod):
        """
        Release key release of the key.

        Args:
            self: (todo): write your description
            k: (todo): write your description
            mod: (todo): write your description
        """
        if k==key.LEFT:  self.move[0] = False
        if k==key.RIGHT: self.move[1] = False
        if k==key.UP:    self.move[2] = False
        if k==key.DOWN:  self.move[3] = False
