import torch as t
from models.noise.generator import OrnsteinUhlenbeckNoiseGen

default_ou_generator = None


def add_uniform_noise_to_action(action: t.Tensor, noise_param=(0.0, 1.0), ratio=1.0):
    """
    Add uniform noise to action tensor.

    Args:
        action: Raw action
        noise_param: Range of the uniform noise. The first element is min, the second element
                     is max.
                     If it is tuple(float, float), then the same uniform noise will be applied
                     to action[*, :].
                     If it is a tuple of tuples, then for each action[*, i] slice i, uniform
                     noise with noise_param[i] will be applied respectively.
        ratio: Sampled noise is multiplied with this ratio before being applied to action

    Returns:
        Action with noise
    """
    if isinstance(noise_param[0], tuple):
        if len(noise_param) != action.shape[-1]:
            raise ValueError("Noise param length doesn't match the last dimension of action")
        noise = t.rand(action.shape, device=action.device)
        for i in range(action.shape[-1]):
            noi_p = noise_param[i]
            noise.view(-1, noise.shape[-1])[:, i] *= noi_p[1] - noi_p[0]
            noise.view(-1, noise.shape[-1])[:, i] += noi_p[0]
    else:
        noise = t.rand(action.shape, device=action.device) \
                * (noise_param[1] - noise_param[0]) + noise_param[0]
    return action + noise * ratio


def add_clipped_normal_noise_to_action(action: t.Tensor, noise_param=(0.0, 1.0, -1.0, 1.0), ratio=1.0):
    """
    Add clipped normal noise to action tensor.

    Args:
        action: Raw action
        noise_param: Param and range of the normal noise. The first two elements are normal
                     noise mean and sigma, the last two elements are min and max noise range.
                     If it is tuple(float, float, float, float), then the same clipped
                     normal noise will be applied to action[*, :].
                     If it is a tuple of tuples, then for each action[*, i] slice i, clipped
                     normal noise with noise_param[i] will be applied respectively.
        ratio: Sampled noise is multiplied with this ratio before being applied to action

    Returns:
        Action with noise
    """
    if isinstance(noise_param[0], tuple):
        if len(noise_param) != action.shape[-1]:
            raise ValueError("Noise param length doesn't match the last dimension of action")
        noise = t.rand(action.shape, device=action.device)
        for i in noise_param(action.shape[-1]):
            noi_p = noise_param[i]
            noise.view(-1, noise.shape[-1])[:, i] *= noi_p[1] - noi_p[0]
            noise.view(-1, noise.shape[-1])[:, i] += noi_p[0]
            noise.view(-1, noise.shape[-1])[:, i].clamp(noi_p[2], noi_p[3])
    else:
        noise = t.rand(action.shape, device=action.device) \
                * (noise_param[1] - noise_param[0]) + noise_param[0]
        noise.clamp(noise_param[2], noise_param[3])
    return action + noise * ratio


def add_normal_noise_to_action(action: t.Tensor, noise_param=(0.0, 1.0), ratio=1.0):
    """
    Add normal noise to action tensor.

    Args:
        action: Raw action
        noise_param: Param of the normal noise. The first element is mean, the second element
                     is sigma.
                     If it is tuple(float, float), then the same uniform noise will be applied
                     to action[*, :].
                     If it is a tuple of tuples, then for each action[*, i] slice i, uniform
                     noise with noise_param[i] will be applied respectively.
        ratio: Sampled noise is multiplied with this ratio before being applied to action

    Returns:
        Action with noise
    """
    if isinstance(noise_param[0], tuple):
        if len(noise_param) != action.shape[-1]:
            raise ValueError("Noise range length doesn't match the last dimension of action")
        noise = t.randn(action.shape, device=action.device)
        for i in range(action.shape[-1]):
            noi_p = noise_param[i]
            noise.view(-1, noise.shape[-1])[:, i] *= noi_p[1]
            noise.view(-1, noise.shape[-1])[:, i] += noi_p[0]
    else:
        noise = t.rand(action.shape, device=action.device) \
                * noise_param[1] + noise_param[0]
    return action + noise * ratio


def add_ou_noise_to_action(action: t.Tensor, noise_param=(0.0, 1.0), ratio=1.0, reset=False):
    """
    Add Ornstein-Uhlenbeck noise to action tensor.
    Note: Ornstein-Uhlenbeck noise generator is shared.

    Args:
        action: Raw action
        noise_param: OrnsteinUhlenbeckGen params
        ratio: Sampled noise is multiplied with this ratio before being applied to action

    Returns:
        Action with noise
    """
    global default_ou_generator
    if reset:
        default_ou_generator = None
    if default_ou_generator is None:
        default_ou_generator = OrnsteinUhlenbeckNoiseGen(action.shape, *noise_param)
        default_ou_generator.reset()
    return action + default_ou_generator() * ratio
