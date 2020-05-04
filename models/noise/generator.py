import torch as t
import torch.distributions as tdist


class NoiseGen(object):
    """
    Base class for noise generators.
    """
    def reset(self):
        pass


class NormalNoiseGen(NoiseGen):
    def __init__(self, shape, mu, sigma):
        """
        Normal noise generator.
        Args:
            mu: Average mean of noise
            sigma: Sigma of the normal distribution.
        """
        self.mu = mu
        self.sigma = sigma
        self.dist = tdist.normal.Normal(mu, sigma)
        self.shape = shape

    def __call__(self):
        return self.dist.sample(self.shape)

    def __repr__(self):
        return 'NormalNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class UniformNoiseGen(NoiseGen):
    def __init__(self, shape, min, max):
        """
        Normal noise generator.
        Args:
            min: Minimum of noise.
            max: Maximum of noise.
        """
        self.min = min
        self.max = max
        self.dist = tdist.uniform.Uniform(min, max)
        self.shape = shape

    def __call__(self):
        return self.dist.sample(self.shape)

    def __repr__(self):
        return 'UniformNoise(min={}, max={})'.format(self.min, self.max)


class OrnsteinUhlenbeckNoiseGen(NoiseGen):
    def __init__(self, shape, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        """
        Ornstein-Uhlenbeck noise generator.
        Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        :math:` X_{n+1} = X_n + \theta (\mu - X_n)\Delta t + \sigma \Delta W_n`
        Args:
            mu: Average mean of noise.
            sigma: Weight of the random wiener process.
            theta: Weight of difference correction
            dt: Time step size.
            x0: Start x value.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = t.tensor(dt) if not isinstance(dt, t.Tensor) else dt
        self.norm_dist = tdist.normal.Normal(loc=0.0, scale=1.0)
        self.shape = shape
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * t.sqrt(self.dt) * self.norm_dist.sample(self.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else t.zeros(self.shape)

    def __repr__(self):
        return 'OrnsteinUhlenbeckNoise(mu={}, sigma={})'.format(self.mu, self.sigma)