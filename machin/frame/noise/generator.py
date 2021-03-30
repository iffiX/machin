from abc import ABC, abstractmethod
from typing import Any
import torch as t
import torch.distributions as tdist

from machin.utils.checker import check_shape


class NoiseGen(ABC):
    """
    Base class for noise generators.
    """

    @abstractmethod
    def __call__(self, device=None):
        """
        Generate a noise tensor, and move it to the specified device.
        """

    @abstractmethod
    def __repr__(self):
        """
        Return a correct representation of the noise distribution, must
        conform to style: "<NoiseName>(param1=..., param2=...)".
        """

    def reset(self):
        """
        Reset internal states of the noise generator, if it has any.
        """


class NormalNoiseGen(NoiseGen):
    def __init__(self, shape: Any, mu: float = 0.0, sigma: float = 1.0):
        """
        Normal noise generator.

        Example:
            >>> gen = NormalNoiseGen([2, 3], 0, 1)
            >>> gen("cuda:0")
            tensor([[-0.5957,  0.2360,  1.0999],
                    [ 1.6259,  1.2052, -0.0667]], device="cuda:0")

        Args:
            shape: Output shape.
            mu: Average mean of normal noise.
            sigma: Standard deviation of normal noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.dist = tdist.normal.Normal(mu, sigma)
        self.shape = shape

    def __call__(self, device=None):
        if device is not None:
            return self.dist.sample(self.shape).to(device)
        else:
            return self.dist.sample(self.shape)

    def __repr__(self):
        return f"NormalNoise(mu={self.mu}, sigma={self.sigma})"


class ClippedNormalNoiseGen(NoiseGen):
    def __init__(
        self,
        shape: Any,
        mu: float = 0.0,
        sigma: float = 1.0,
        nmin: float = -1.0,
        nmax: float = 1.0,
    ):
        """
        Normal noise generator.

        Example:
            >>> gen = NormalNoiseGen([2, 3], 0, 1)
            >>> gen("cuda:0")
            tensor([[-0.5957,  0.2360,  1.0999],
                    [ 1.6259,  1.2052, -0.0667]], device="cuda:0")

        Args:
            shape: Output shape.
            mu: Average mean of normal noise.
            sigma: Standard deviation of normal noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.dist = tdist.normal.Normal(mu, sigma)
        self.min = nmin
        self.max = nmax
        self.shape = shape

    def __call__(self, device=None):
        if device is not None:
            return self.dist.sample(self.shape).to(device)
        else:
            return self.dist.sample(self.shape)

    def __repr__(self):
        return (
            f"ClippedNormalNoise(mu={self.mu}, sigma={self.sigma},"
            f" min={self.min}, max={self.max})"
        )


class UniformNoiseGen(NoiseGen):
    def __init__(self, shape: Any, umin: float = 0.0, umax: float = 1.0):
        """
        Normal noise generator.

        Example:
            >>> gen = UniformNoiseGen([2, 3], 0, 1)
            >>> gen("cuda:0")
            tensor([[0.0745, 0.6581, 0.9572],
                    [0.4450, 0.8157, 0.6421]], device="cuda:0")

        Args:
            shape: Output shape.
            umin: Minimum value of uniform noise.
            umax: Maximum value of uniform noise.
        """
        self.min = umin
        self.max = umax
        self.dist = tdist.uniform.Uniform(umin, umax)
        self.shape = shape

    def __call__(self, device=None):
        if device is not None:
            return self.dist.sample(self.shape).to(device)
        else:
            return self.dist.sample(self.shape)

    def __repr__(self):
        return f"UniformNoise(min={self.min}, max={self.max})"


class OrnsteinUhlenbeckNoiseGen(NoiseGen):
    def __init__(
        self,
        shape: Any,
        mu: float = 0.0,
        sigma: float = 1.0,
        theta: float = 0.15,
        dt: float = 1e-2,
        x0: t.Tensor = None,
    ):
        """
        Ornstein-Uhlenbeck noise generator.
        Based on `definition <http://math.stackexchange.com/questions\
/1287634/implementing-ornstein-uhlenbeck-in-matlab>`_:

        :math:`X_{n+1} = X_n + \\theta (\\mu - X_n)\\Delta t + \\sigma \
        \\Delta W_n`

        Example:
            >>> gen = OrnsteinUhlenbeckNoiseGen([2, 3], 0, 1)
            >>> gen("cuda:0")
            tensor([[ 0.1829,  0.1589, -0.1932],
                    [-0.1568,  0.0579,  0.2107]], device="cuda:0")
            >>> gen.reset()


        Args:
            shape: Output shape.
            mu: Average mean of noise.
            sigma: Weight of the random wiener process.
            theta: Weight of difference correction.
            dt: Time step size.
            x0: Initial x value. Must have the same shape as ``shape``.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = t.tensor(dt) if not isinstance(dt, t.Tensor) else dt
        self.norm_dist = tdist.normal.Normal(loc=0.0, scale=1.0)
        self.shape = shape
        self.x0 = x0
        if x0 is not None:
            check_shape(x0, list(shape))
        self.x_prev = None
        self.reset()

    def __call__(self, device=None):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * t.sqrt(self.dt) * self.norm_dist.sample(self.shape)
        )
        self.x_prev = x
        if device is not None:
            return x.to(device)
        else:
            return x

    def reset(self):
        """
        Reset the generator to its initial state.
        """
        self.x_prev = self.x0 if self.x0 is not None else t.zeros(self.shape)

    def __repr__(self):
        return f"OrnsteinUhlenbeckNoise(mu={self.mu}, sigma={self.sigma})"
