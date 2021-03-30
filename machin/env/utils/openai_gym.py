def disable_view_window(display=None):
    """
    Args:
        display: The X display name to use, by default it creates a new display.
            you can set it to ":0" to use existing X displays.
    """
    # Disable pop up windows and render in background
    # by injecting custom viewer constructor.
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, display=display, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor
