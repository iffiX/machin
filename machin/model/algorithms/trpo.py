from ..nets import NeuralNetworkModule
from machin.frame.algorithms.trpo import TRPO
from torch.distributions import Normal, Categorical
import torch as t
import torch.nn as nn


class ActorDiscrete(NeuralNetworkModule):
    def __init__(self):
        super().__init__()
        self.action_param = None

    def sample(self, probability: t.tensor, action=None):
        """
        You must call this function to sample an action and its log probability
        during forward().

        Args:
            probability: Probability tensor of shape ``[batch, action_num]``,
                usually produced by a softmax layer.
            action: The action to be evaluated. set to ``None`` if you are sampling
                a new batch of actions.

        Returns:
            Action tensor of shape ``[batch, 1]``,
            Action log probability tensor of shape ``[batch, 1]``.
        """
        batch_size = probability.shape[0]
        # dx (xlnx) = lnx + 1, x must > 0
        self.action_param = probability + 1e-6
        dist = Categorical(probs=probability)
        if action is None:
            action = dist.sample()

        # since categorical sample returns a flat tensor, we need to reshape it.
        return (
            action.view(batch_size, 1),
            dist.log_prob(action.flatten()).view(batch_size, 1),
        )

    def get_kl(self, *args, **kwargs):
        self.forward(*args, **kwargs)
        action_prob1 = self.action_param
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (t.log(action_prob0 / action_prob1))
        return kl.sum(1, keepdim=True)

    def compare_kl(self, params: t.tensor, *args, **kwargs):
        with t.no_grad():
            new_params = TRPO.get_flat_params(self)

            TRPO.set_flat_params(self, params)
            self.forward(*args, **kwargs)
            action_prob_old = self.action_param

            TRPO.set_flat_params(self, new_params)
            self.forward(*args, **kwargs)
            action_prob_new = self.action_param

            kl = action_prob_old * (t.log(action_prob_old) - t.log(action_prob_new))
            return kl.sum(1).mean().item()

    def get_fim(self, *args, **kwargs):
        self.forward(*args, **kwargs)
        # 1 / prob for each discrete action
        # see <<Fisher information matrix for Gaussian and categorical distributions>>
        # https://www.ii.pwr.edu.pl/~tomczak/PDF/[JMT]Fisher_inf.pdf
        M = self.action_param.view(-1).detach()
        return M, self.action_param


class ActorContinuous(NeuralNetworkModule):
    def __init__(self, action_dim, log_std=0):
        super().__init__()
        self.action_param = None
        self.action_log_std = nn.Parameter(t.full([1, action_dim], float(log_std)))

    def sample(self, mean: t.tensor, action=None):
        """
        You must call this function to sample an action and its log probability
        during forward().

        Args:
            mean: Probability tensor of shape ``[batch, action_num]``,
                usually produced by a softmax layer.
            action: The action to be evaluated. set to ``None`` if you are sampling
                a new batch of actions.

        Returns:
            Action tensor of shape ``[batch, action_dim]``,
            Action log probability tensor of shape ``[batch, 1]``.
        """
        self.action_param = mean
        dist = Normal(loc=mean, scale=t.exp(self.action_log_std))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(dim=1, keepdims=True)

    def get_kl(self, *args, **kwargs):
        self.forward(*args, **kwargs)
        mean1 = self.action_param
        mean0 = mean1.detach()
        log_std1 = self.action_log_std.expand_as(mean1)
        log_std0 = log_std1.detach()
        std1 = t.exp(log_std1)
        std0 = std1.detach()
        # KL divergence between two gaussians
        # https://stats.stackexchange.com/questions/7440/kl-divergence-
        # between-two-univariate-gaussians
        kl = (
            log_std1
            - log_std0
            + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2))
            - 0.5
        )
        return kl.sum(1, keepdim=True)

    def compare_kl(self, params: t.tensor, *args, **kwargs):
        with t.no_grad():
            new_params = TRPO.get_flat_params(self)

            TRPO.set_flat_params(self, params)
            self.forward(*args, **kwargs)
            mean1 = self.action_param
            log_std1 = self.action_log_std.expand_as(mean1)
            std1 = t.exp(log_std1)

            TRPO.set_flat_params(self, new_params)
            self.forward(*args, **kwargs)
            mean0 = self.action_param
            log_std0 = self.action_log_std.expand_as(mean1)
            std0 = t.exp(log_std1)

            kl = (
                log_std1
                - log_std0
                + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2))
                - 0.5
            )
            return kl.sum(1).mean().item()

    def get_fim(self, *args, **kwargs):
        self.forward(*args, **kwargs)
        batch_size = self.action_param.shape[0]
        # 1/ sigma^2
        # see <<Fisher information matrix for Gaussian and categorical distributions>>
        # https://www.ii.pwr.edu.pl/~tomczak/PDF/[JMT]Fisher_inf.pdf
        M = t.exp(-2 * self.action_log_std).squeeze(0).repeat(batch_size).detach()
        return M, self.action_param
