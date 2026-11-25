import math

import torch
from torch.distributions import (
    Distribution,
    constraints,
)
from torch.nn.functional import pad, softplus

LOG_2PI = math.log(2 * math.pi)
EPS = 1e-6
__all__ = ["AdditiveLogisticNormal", "FastBeta", "FastDirichlet", "LogitNormal"]


def _lbeta(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def _normal_log_prob(z, loc, scale):
    return -0.5 * ((z - loc) / scale) ** 2 - scale.log() - 0.5 * LOG_2PI


def _sample_z(loc, scale, sample_shape):
    return loc + scale * torch.randn(sample_shape + loc.shape, dtype=loc.dtype, device=loc.device)


class AdditiveLogisticNormal(Distribution):
    """x = softmax([z, 0]), z ~ Normal(loc, scale); число компонент x на 1 больше, чем у loc."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = torch.broadcast_tensors(loc, scale)
        super().__init__(self.loc.shape[:-1], (self.loc.shape[-1] + 1,), validate_args=validate_args)

    # def rsample(self, sample_shape=torch.Size()):
    #     eps = torch.randn(sample_shape + self.loc.shape, dtype=self.loc.dtype, device=self.loc.device)
    #     z = self.loc + self.scale * eps
    #     x = torch.softmax(torch.cat((z, torch.zeros_like(z[..., :1])), dim=-1), dim=-1)
    #     x = x.clamp_min(EPS)
    #     return x / x.sum(dim=-1, keepdim=True)

    def rsample(self, sample_shape=torch.Size()):
        z = _sample_z(self.loc, self.scale, sample_shape)
        x = pad(z, (0, 1)).softmax(dim=-1).clamp_min(EPS)
        return x / x.sum(dim=-1, keepdim=True)

    # def log_prob(self, value):
    #     log_x = value.log()
    #     z = log_x[..., :-1] - log_x[..., -1:]
    #     normal = -0.5 * ((z - self.loc) / self.scale) ** 2 - self.scale.log() - 0.5 * LOG_2PI
    #     return normal.sum(-1) - log_x.sum(-1)

    def log_prob(self, value):
        log_x = value.log()
        z = log_x[..., :-1] - log_x[..., -1:]
        return _normal_log_prob(z, self.loc, self.scale).sum(-1) - log_x.sum(-1)

    def entropy(self):
        z = _sample_z(self.loc, self.scale, torch.Size())
        log_x = pad(z, (0, 1)).log_softmax(dim=-1)
        return -(_normal_log_prob(z, self.loc, self.scale).sum(-1) - log_x.sum(-1))

    # def entropy(self):
    #     return -self.log_prob(self.rsample())


class LogitNormal(Distribution):
    """x = sigmoid(z), z ~ Normal(loc, scale), x ∈ (0, 1)."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = torch.broadcast_tensors(loc, scale)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        return (_sample_z(self.loc, self.scale, sample_shape)).sigmoid().clamp(EPS, 1 - EPS)

    def log_prob(self, value):
        log_x, log_1mx = value.log(), (-value).log1p()
        return _normal_log_prob(log_x - log_1mx, self.loc, self.scale) - log_x - log_1mx

    # def entropy(self):
    #     z = _sample_z(self.loc, self.scale, torch.Size())
    #     return -(_normal_log_prob(z, self.loc, self.scale) + softplus(-z) + softplus(z))
    def entropy(self):
        return 0.5 + 0.5 * LOG_2PI + self.scale.log()
