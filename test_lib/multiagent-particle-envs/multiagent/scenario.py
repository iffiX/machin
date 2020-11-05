import numpy as np

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        """
        Creates a new world.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        """
        Reset a world.

        Args:
            self: (todo): write your description
            world: (todo): write your description
        """
        raise NotImplementedError()
