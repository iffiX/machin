from typing import Tuple, Iterable, Union, Dict, Any
import torch as t

from .generator import OrnsteinUhlenbeckNoiseGen


DEFAULT_OU_GEN = None

NoiseParam = Union[Iterable[Tuple], Tuple]


def add_uniform_noise_to_action(
    action: t.Tensor, noise_param: NoiseParam = (0.0, 1.0), ratio: float = 1.0
):
    """
    Add uniform noise to action tensor.

    Hint:
        The innermost tuple contains: ``(uniform_min, uniform_max)``

        If ``noise_param`` is ``Tuple[float, float]``, then the same uniform
        noise will be added to ``action[*, :]``.

        If ``noise_param`` is ``Iterable[Tuple[float, float]]``, then for each
        ``action[*, i]`` slice i, uniform noise with ``noise_param[i]`` will
        be added respectively.

    Args:
        action: Raw action.
        noise_param: Param of the uniform noise.
        ratio: Sampled noise is multiplied with this ratio.

    Returns:
        Action with uniform noise.
    """
    if isinstance(noise_param[0], tuple):
        if len(noise_param) != action.shape[-1]:
            raise ValueError(
                "Noise param length doesn't match " "the last dimension of action"
            )
        noise = t.rand(action.shape, device=action.device)
        for i in range(action.shape[-1]):
            noi_p = noise_param[i]
            noise.view(-1, noise.shape[-1])[:, i] *= noi_p[1] - noi_p[0]
            noise.view(-1, noise.shape[-1])[:, i] += noi_p[0]
    else:
        noise_param = noise_param  # type: Tuple[float, float]
        noise = (
            t.rand(action.shape, device=action.device)
            * (noise_param[1] - noise_param[0])
            + noise_param[0]
        )
    return action + noise * ratio


def add_clipped_normal_noise_to_action(
    action: t.Tensor, noise_param: NoiseParam = (0.0, 1.0, -1.0, 1.0), ratio=1.0
):
    """
    Add clipped normal noise to action tensor.

    Hint:
        The innermost tuple contains:
        ``(normal_mean, normal_sigma, clip_min, clip_max)``

        If ``noise_param`` is ``Tuple[float, float, float, float]``,
        then the same clipped normal noise will be added to ``action[*, :]``.

        If ``noise_param`` is ``Iterable[Tuple[float, float, float, float]]``,
        then for each ``action[*, i]`` slice i, clipped normal noise with
        ``noise_param[i]`` will be applied respectively.

    Args:
        action: Raw action
        noise_param: Param of the normal noise.
        ratio: Sampled noise is multiplied with this ratio.

    Returns:
        Action with uniform noise.
    """
    if isinstance(noise_param[0], tuple):
        if len(noise_param) != action.shape[-1]:
            raise ValueError(
                "Noise param length doesn't match " "the last dimension of action"
            )
        noise = t.rand(action.shape, device=action.device)
        for i in range(action.shape[-1]):
            noi_p = noise_param[i]
            noise.view(-1, noise.shape[-1])[:, i] *= noi_p[1] - noi_p[0]
            noise.view(-1, noise.shape[-1])[:, i] += noi_p[0]
            noise.view(-1, noise.shape[-1])[:, i].clamp(noi_p[2], noi_p[3])
    else:
        noise_param = noise_param  # type: Tuple[float, float, float, float]
        noise = (
            t.rand(action.shape, device=action.device)
            * (noise_param[1] - noise_param[0])
            + noise_param[0]
        )
        noise.clamp(noise_param[2], noise_param[3])
    return action + noise * ratio


def add_normal_noise_to_action(action: t.Tensor, noise_param=(0.0, 1.0), ratio=1.0):
    """
    Add normal noise to action tensor.

    Hint:
        The innermost tuple contains:
        ``(normal_mean, normal_sigma)``

        If ``noise_param`` is ``Tuple[float, float]``,
        then the same normal noise will be added to ``action[*, :]``.

        If ``noise_param`` is ``Iterable[Tuple[float, float]]``,
        then for each ``action[*, i]`` slice i, clipped normal noise with
        ``noise_param[i]`` will be applied respectively.

    Args:
        action: Raw action
        noise_param: Param of the normal noise.
        ratio: Sampled noise is multiplied with this ratio.

    Returns:
        Action with normal noise.
    """
    if isinstance(noise_param[0], tuple):
        if len(noise_param) != action.shape[-1]:
            raise ValueError(
                "Noise param length doesn't match " "the last dimension of action"
            )
        noise = t.randn(action.shape, device=action.device)
        for i in range(action.shape[-1]):
            noi_p = noise_param[i]
            noise.view(-1, noise.shape[-1])[:, i] *= noi_p[1]
            noise.view(-1, noise.shape[-1])[:, i] += noi_p[0]
    else:
        noise = (
            t.rand(action.shape, device=action.device) * noise_param[1] + noise_param[0]
        )
    return action + noise * ratio


def add_ou_noise_to_action(
    action: t.Tensor, noise_param: Dict[str, Any] = None, ratio=1.0, reset=False
):
    """
    Add Ornstein-Uhlenbeck noise to action tensor.

    Warning:
        Ornstein-Uhlenbeck noise generator is shared. And you cannot
        specify OU noise of different distributions
        for each of the last dimension of your action.

    Args:
        action: Raw action
        noise_param: :class:`.OrnsteinUhlenbeckGen` params. Used as
            keyword arguments of the generator. Will only be effective if
            ``reset`` is ``True``.
        ratio: Sampled noise is multiplied with this ratio.
        reset: Whether to reset the default Ornstein-Uhlenbeck noise generator.

    Returns:
        Action with Ornstein-Uhlenbeck noise.
    """
    global DEFAULT_OU_GEN
    if reset:
        DEFAULT_OU_GEN = None
    if DEFAULT_OU_GEN is None:
        DEFAULT_OU_GEN = OrnsteinUhlenbeckNoiseGen(action.shape, **noise_param)
        DEFAULT_OU_GEN.reset()
    return action + DEFAULT_OU_GEN(action.device) * ratio
