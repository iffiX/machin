def disable_view_window():
    # Disable pop up windows and render in background
    # by injecting custom viewer constructor.
    from gym.envs.classic_control import rendering

    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor
