import numpy as np
import torch

def softmax(x, axis=0):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=axis)


def softplus(x):
    return np.log(1.0 + np.exp(x))


def inv_softplus(x):
    return np.log(np.exp(x) - 1)

def log_sum_exp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def gamma_like(dT, ab):
    """
    Calculates gamma likelihood for a set of arrival times and
    gamma parameters for a set of discrete states
    Args:
        dT (): Interarrival times (2D tensor where second D is size 1)
        ab (): Gamma parameters (N x K x 2 tensor)

    Returns:

    """
    return ab[:, :, 0] * torch.log(ab[:, :, 1]) - torch.lgamma(ab[:, :, 0]) + \
           torch.log(dT) * (ab[:, :, 0] - 1) - dT * ab[:, :, 1]