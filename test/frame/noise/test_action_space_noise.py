from machin.frame.noise.action_space_noise import (
    add_normal_noise_to_action,
    add_clipped_normal_noise_to_action,
    add_uniform_noise_to_action,
    add_ou_noise_to_action
)

import pytest
import torch as t


class TestAllActionSpaceNoise(object):
    ########################################################################
    # Test for add_normal_noise_to_action
    ########################################################################
    param_add_normal_noise_to_action = [
        (t.zeros([5, 2]), (0.0, 1.0), None, None),
        (t.zeros([5, 2]), ((0.0, 1.0), (0.0, 1.0)), None, None),
        (t.zeros([5, 2]), ((0.0, 1.0),), ValueError,
         "Noise param length doesn't match")
    ]

    @pytest.mark.parametrize("action,noise_param,exception,match",
                             param_add_normal_noise_to_action)
    def test_add_normal_noise_to_action(self, action, noise_param,
                                        exception, match, pytestconfig):
        """
        Add an action to an action.

        Args:
            self: (todo): write your description
            action: (str): write your description
            noise_param: (float): write your description
            exception: (todo): write your description
            match: (todo): write your description
            pytestconfig: (todo): write your description
        """
        if exception is not None:
            with pytest.raises(exception, match=match):
                add_normal_noise_to_action(action, noise_param)
        else:
            add_normal_noise_to_action(action, noise_param)
            add_normal_noise_to_action(
                action.to(pytestconfig.getoption("gpu_device")), noise_param
            )

    ########################################################################
    # Test for add_clipped_normal_noise_to_action
    ########################################################################
    param_add_clipped_normal_noise_to_action = [
        (t.zeros([5, 2]), (0.0, 1.0, -1.0, 1.0), None, None),
        (t.zeros([5, 2]), ((0.0, 1.0, -1.0, 1.0),
                           (0.0, 1.0, -0.5, 0.5)), None, None),
        (t.zeros([5, 2]), ((0.0, 1.0, -1.0, 1.0),), ValueError,
         "Noise param length doesn't match")
    ]

    @pytest.mark.parametrize("action,noise_param,exception,match",
                             param_add_clipped_normal_noise_to_action)
    def test_add_clipped_normal_noise_to_action(self, action, noise_param,
                                                exception, match, pytestconfig):
        """
        Add an action to an clipped action.

        Args:
            self: (todo): write your description
            action: (str): write your description
            noise_param: (todo): write your description
            exception: (todo): write your description
            match: (todo): write your description
            pytestconfig: (todo): write your description
        """
        if exception is not None:
            with pytest.raises(exception, match=match):
                add_clipped_normal_noise_to_action(action, noise_param)
        else:
            add_clipped_normal_noise_to_action(action, noise_param)
            add_normal_noise_to_action(
                action.to(pytestconfig.getoption("gpu_device")), noise_param
            )

    ########################################################################
    # Test for add_uniform_noise_to_action
    ########################################################################
    param_add_uniform_noise_to_action = [
        (t.zeros([5, 2]), (0.0, 1.0), None, None),
        (t.zeros([5, 2]), ((0.0, 1.0), (0.0, 1.0)), None, None),
        (t.zeros([5, 2]), ((0.0, 1.0),), ValueError,
         "Noise param length doesn't match")
    ]

    @pytest.mark.parametrize("action,noise_param,exception,match",
                             param_add_uniform_noise_to_action)
    def test_add_uniform_noise_to_action(self, action, noise_param,
                                         exception, match, pytestconfig):
        """
        Add an action to an action.

        Args:
            self: (todo): write your description
            action: (str): write your description
            noise_param: (float): write your description
            exception: (todo): write your description
            match: (todo): write your description
            pytestconfig: (todo): write your description
        """
        if exception is not None:
            with pytest.raises(exception, match=match):
                add_uniform_noise_to_action(action, noise_param)
        else:
            add_uniform_noise_to_action(action, noise_param)
            add_normal_noise_to_action(
                action.to(pytestconfig.getoption("gpu_device")), noise_param
            )

    ########################################################################
    # Test for add_ou_noise_to_action
    ########################################################################
    param_add_ou_noise_to_action = [
        (t.zeros([5, 2]), {"x0": t.ones([5, 2])}, True),
        (t.zeros([5, 2]), {"x0": t.ones([5, 2])}, False),
    ]

    @pytest.mark.parametrize("action,noise_param,reset",
                             param_add_ou_noise_to_action)
    def test_add_ou_noise_to_action(self, action, noise_param,
                                    reset, pytestconfig):
        """
        Add noise to the i / oise_noise. e.

        Args:
            self: (todo): write your description
            action: (str): write your description
            noise_param: (float): write your description
            reset: (todo): write your description
            pytestconfig: (todo): write your description
        """
        add_ou_noise_to_action(action, noise_param, reset=reset)
        add_ou_noise_to_action(
            action.to(pytestconfig.getoption("gpu_device")),
            noise_param, reset=reset
        )
