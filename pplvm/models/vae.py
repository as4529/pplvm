import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from tqdm import trange, tqdm

class MixtureVAE(object):

    def __init__(self, encoder, decoder, D_h, K, means=None, variances=None):
        """

        :param encoder:
        :type encoder:
        :param decoder:
        :type decoder:
        :param D_h:
        :type D_h:
        """

        self.encoder = encoder
        self.decoder = decoder
        self.D_h = D_h
        self.K = K
        if means is None:
            self.prior_mus = [torch.tensor(torch.randn(self.D_h) * 1e-2,
                                           dtype=torch.float64, requires_grad=True)
                                for k in range(self.K)]
        else:
            self.prior_mus = means
        if variances is None:
            self.prior_sigmas = [torch.tensor(np.eye(self.D_h),
                                             dtype=torch.float64,
                                             requires_grad=True)
                                 for k in range(self.K)]
        else:
            self.prior_sigmas = variances
        self.pi = torch.ones(self.K, dtype=torch.float64, requires_grad=True)
        self.cluster_params = self.prior_mus + self.prior_sigmas + [self.pi]
        self.network_params = list(self.encoder.parameters()) +\
                              list(self.decoder.parameters())
        self.params = [self.pi] + self.cluster_params + self.network_params

    def elbo(self, x):

        q = self.encode(x)
        h = q.rsample()
        loglike = self.decode(h).log_prob(x).sum()
        priorprob = self.prior_prob(h).logsumexp(dim=1).sum()
        ent = q.entropy().sum()
        elbo = loglike + priorprob + ent

        return elbo

    def prior_prob(self, h):

        self.pi_norm = torch.nn.LogSoftmax(dim=0)(self.pi)
        prob = []
        for k in range(self.K):
            p = MultivariateNormal(self.prior_mus[k],
                                   scale_tril=self.prior_sigmas[k].tril())
            prob.append(p.log_prob(h) + self.pi_norm[k])
        prob = torch.stack(prob)

        return prob.t()

    def mixture_prob(self, h):

        prob = []
        for k in range(self.K):
            p = MultivariateNormal(self.prior_mus[k],
                                   scale_tril=self.prior_sigmas[k].tril())
            prob.append(p.log_prob(h))
        prob = torch.stack(prob)

        return prob.t()

    def encode(self, x):

        return Normal(*self.encoder(x))

    def decode(self, h):

        return Normal(*self.decoder(h))

    def max_z(self, x):

        h = self.encoder(x)[0]
        prob = self.prior_prob(h)
        max = torch.argmax(prob, dim=1).detach().numpy()

        return max



