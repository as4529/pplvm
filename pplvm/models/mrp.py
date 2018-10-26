from pplvm.utils.utils import log_sum_exp
import torch
from torch.optim import Adam
from torch.distributions import MultivariateNormal, Gamma
import torch.nn.functional as F
from pplvm.utils.utils import gamma_like
from torch.autograd import Function
from scipy.stats import truncnorm
from pplvm.message_passing.normalizer import hmmnorm_cython
import numpy as np
from tqdm import trange, tqdm_notebook
from copy import copy


class MRP:

    def __init__(self,
                 likelihood,
                 pi=None,
                 A=None,
                 log_ab=None):

        self.likelihood = likelihood
        self.K = self.likelihood.K
        self.pi = pi if pi is not None else \
            torch.ones(self.K, dtype=torch.float64, requires_grad=True)
        self.A = A if A is not None else \
            torch.ones(self.K, self.K, dtype=torch.float64, requires_grad=True)
        self.log_ab = log_ab if log_ab is not None else \
            torch.randn(self.K, 2, dtype=torch.float64, requires_grad=True)
        self.model_params = [self.pi, self.A, self.log_ab]

    def log_like(self, Y, dT):

        prob = []
        for k in range(self.K):
            p_g = Gamma(torch.exp(self.log_ab[k][0]), torch.exp(self.log_ab[k][1]))
            prob_g = p_g.log_prob(dT)
            prob.append(prob_g)
        prob = torch.stack(prob).t() + self.likelihood.mixture_prob(Y)

        return prob

    def loss(self, Y, dT):

        ll = self.log_like(Y, dT)
        N = len(dT)

        A_expand = torch.nn.LogSoftmax(dim=1)(self.A).expand(N - 1,
                                                             self.K, self.K)
        pi0_norm = torch.nn.LogSoftmax(dim=0)(self.pi)

        loss = - 1. * hmmnorm_cython(pi0_norm,
                                     A_expand.contiguous(),
                                     ll.contiguous())

        return loss

    def max_z(self, Y, dT):

        ll = self.log_like(Y, dT)
        N = len(dT)
        A_expand = torch.nn.LogSoftmax(dim=1)(self.A).expand(N - 1,
                                                             self.K, self.K)
        pi0_norm = torch.nn.LogSoftmax()(self.pi)
        T, K = ll.shape
        delta = pi0_norm + ll[0]
        idx = []
        for t in range(1, T):
            max_idx = torch.max(delta + A_expand[t - 1].t(), dim=1)
            idx.append(max_idx[1])
            delta = max_idx[0] + ll[t]
        i = torch.argmax(delta)
        Z = [i]
        for t in range(T - 2, -1, -1):
            i = idx[t][i]
            Z.append(i)
        return list(reversed(Z))