import torch as t
from models.noise.generator import OrnsteinUhlenbeckNoiseGen

default_ou_generator = None


def add_uniform_noise_to_action(action: t.Tensor, noise_range=(0.0, 1.0), ratio=1.0):
    """
    Add uniform noise to action tensor.

    Args:
        action: Raw action
        noise_range: Range of the uniform noise. If it is tuple(float, float), then the same
                     uniform noise will be applied to action[*, :]. If it is a tuple of tuples,
                     then for each action[*, i] slice i, uniform noise with noise_range[i]
                     will be applied respectively.
        ratio: Sampled noise is multiplied with this ratio before being applied to action

    Returns:
        Action with noise
    """
    if isinstance(noise_range[0], tuple):
        if len(noise_range) != action.shape[-1]:
            raise ValueError("Noise range length doesn't match the last dimension of action")
        noise = t.rand(action.shape, device=action.device)
        for i in range(action.shape[-1]):
            noi_r = noise_range[i]
            noise.view(-1, noise.shape[-1])[:, i] *= noi_r[1] - noi_r[0]
            noise.view(-1, noise.shape[-1])[:, i] += noi_r[0]
    else:
        noise = t.rand(action.shape, device=action.device) \
                * (noise_range[1] - noise_range[0]) + noise_range[0]
    return action + noise * ratio


def add_normal_noise_to_action(action: t.Tensor, noise_param=(0.0, 1.0), ratio=1.0):
    """
    Add normal noise to action tensor.

    Args:
        action: Raw action
        noise_range: Range of the uniform noise. If it is tuple(float, float), then the same
                     normal noise will be applied to action[*, :]. If it is a tuple of tuples,
                     then for each action[*, i] slice i, normal noise with noise_param[i]
                     will be applied respectively.
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
