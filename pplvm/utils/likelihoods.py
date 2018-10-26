import numpy as np
import torch
from torch.distributions import Normal, MultivariateNormal
from scipy.stats import truncnorm


class MixtureGaussian:

    def __init__(self, K, D, means=None, variances=None):

        self.K = K
        self.D = D
        if means is None:
            self.prior_mus = [torch.zeros(self.D,
                                          dtype=torch.float64, requires_grad=True)
                              for k in range(self.K)]
        else:
            self.prior_mus = means
        if variances is None:
            self.prior_sigmas = [torch.tensor(np.ones(self.D) * -1,
                                              dtype=torch.float64, requires_grad=True)
                                 for k in range(self.K)]
        self.params = self.prior_mus + self.prior_sigmas


    def mixture_prob(self, Y):

        prob = []
        for k in range(self.K):
            p = Normal(self.prior_mus[k], self.prior_sigmas[k].exp())
            prob.append(p.log_prob(Y).sum(dim=1))
        prob = torch.stack(prob)

        return prob.t()

class MixtureMultivariateGaussian:

    def __init__(self, K, D, means=None, variances=None):

        self.K = K
        self.D = D
        if means is None:
            self.prior_mus = [torch.zeros(self.D,
                                          dtype=torch.float64, requires_grad=True)
                              for k in range(self.K)]
        else:
            self.prior_mus = means
        if variances is None:
            self.prior_sigmas = [torch.eye(self.D,
                                           dtype=torch.float64, requires_grad=True)
                                 for k in range(self.K)]
        self.params = self.prior_mus + self.prior_sigmas


    def mixture_prob(self, Y):

        prob = []
        for k in range(self.K):
            p = MultivariateNormal(self.prior_mus[k],
                                   scale_tril=self.prior_sigmas[k].tril())
            prob.append(p.log_prob(Y))
        prob = torch.stack(prob)

        return prob.t()