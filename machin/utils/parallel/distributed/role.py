class RoleBase(object):
    """
    Base of all roles, derive from this class to implement your own roles.

    :ivar NAME: A unique name of this role.
    """
    NAME = ""

    def __init__(self, index):
        """
        Args:
            index: worker index of this role, e.g.: ``"some_role:10"`` should
                have ``10`` as index.
        """
        self.role_index = index

    def on_init(self):
        """
        :meth:`on_init` is executed before the process executes your event loop.
        """
        pass

    def on_stop(self):
        """
        :meth:`on_init` is executed after the process stops your event loop and
        exit
        """
        pass

    def main(self):
        """
        :meth:`main` Is the main event loop of the role
        """
        pass

    def __str__(self):
        return self.NAME + ":" + str(self.role_index)
