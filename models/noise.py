import torch as t
import torch.distributions as tdist
from utils.logging import default_logger as logger


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_dev(self):
        return self.current_stddev

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class Noise(object):
    """
    Base class for noise generators.
    """
    def reset(self):
        pass


class NormalNoise(Noise):
    def __init__(self, mu, sigma):
        """
        Normal noise generator.
        Args:
            mu: Average mean of noise
            sigma: Sigma of the normal distribution.
        """
        self.mu = mu
        self.sigma = sigma
        self.dist = tdist.normal.Normal(mu, sigma)

    def __call__(self, shape):
        return self.dist.sample(shape)

    def __repr__(self):
        return 'NormalNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class OrnsteinUhlenbeckNoise(Noise):
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


def gen_perturb_hook(module, perturb_switch, reset_switch, perturb_gen):
    org_params = {}
    noisy_params = {}

    def perturb_pre_hook(*_):
        with t.no_grad():
            if perturb_switch.get():
                if len(noisy_params) != 0 and not reset_switch.get():
                    # use generated noisy parameters
                    for param_name, param_value in module.named_parameters():
                        param_value.set_(noisy_params[param_name])
                else:
                    org_params.clear()
                    noisy_params.clear()
                    for param_name, param_value in module.named_parameters():
                        org_params[param_name] = param_value.clone()
                        param_value += perturb_gen(param_value.shape)
                        noisy_params[param_name] = param_value.clone()
            elif not perturb_switch.get():
                # use original paramters
                if len(org_params) != 0:
                    for param_name, param_value in module.named_parameters():
                        param_value.set_(org_params[param_name])

    def perturb_post_hook(*_):
        # called before backward update
        with t.no_grad():
            if len(org_params) != 0:
                for param_name, param_value in module.named_parameters():
                    param_value.set_(org_params[param_name])

    return (perturb_pre_hook, perturb_post_hook)


def perturb_model(model, device, perturb_switch, reset_switch,
                  distance_func=lambda x, y: t.dist(x, y, 2),
                  desired_action_stddev=0.5, noise_distribution=NormalNoise(0, 1.0)):
    tmp_action = {}
    post_hooks = []
    param_noise_spec = AdaptiveParamNoiseSpec(desired_action_stddev=desired_action_stddev)
    param_noise_dist = noise_distribution
    param_noise_gen = lambda shape: param_noise_dist(shape).to(device) * param_noise_spec.get_dev()

    def perturb_adjust_hook(model, input, output):
        if perturb_switch.get():
            tmp_action["with_noise"] = output.clone()
        else:
            tmp_action["without_noise"] = output.clone()
        if "with_noise" in tmp_action and "without_noise" in tmp_action:
            # compute l2 distance
            with t.no_grad():
                dist = distance_func(tmp_action["with_noise"], tmp_action["without_noise"])
                tmp_action.clear()
                param_noise_spec.adapt(dist)
                logger.info("Current output distance: {}".format(dist))
                logger.info("Current param noise stddev: {}".format(param_noise_spec.get_dev()))

    model.register_forward_hook(perturb_adjust_hook)

    for sub_name, sub_module in model.named_modules():
        if len([i for i in sub_module.modules()]) != 1:
            continue
        pre_f, post = gen_perturb_hook(sub_module, perturb_switch, reset_switch, param_noise_gen)
        sub_module.register_forward_pre_hook(pre_f)
        post_hooks.append(post)

    def reset():
        for h in post_hooks:
            h()

    return reset