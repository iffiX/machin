class Role(object):
    def __init__(self, index):
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
