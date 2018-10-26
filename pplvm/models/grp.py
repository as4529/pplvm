import torch
from torch.distributions import Gamma

class GRP:

    def __init__(self,
                 likelihood,
                 pi=None,
                 log_ab=None):

        self.likelihood = likelihood
        self.K = self.likelihood.K
        self.pi = pi if pi is not None else \
            torch.ones(self.K, dtype=torch.float64, requires_grad=True)
        self.log_ab = log_ab if log_ab is not None else \
            torch.randn(self.K, 2, dtype=torch.float64, requires_grad=True)
        self.model_params = [self.pi, self.log_ab]

    def log_like(self, Y, dT):

        prob = []
        self.pi_norm = torch.nn.LogSoftmax(dim=0)(self.pi)
        for k in range(self.K):
            p_g = Gamma(torch.exp(self.log_ab[k][0]), torch.exp(self.log_ab[k][1]))
            prob_g = p_g.log_prob(dT)
            prob.append(prob_g + self.pi_norm[k])
        prob = torch.stack(prob).t() + self.likelihood.mixture_prob(Y)

        return prob

    def loss(self, Y, dT):

        ll = self.log_like(Y, dT)

        return -ll.logsumexp(dim=1).sum()


class GRPVAE:

    def __init__(self,
                 vae,
                 pi=None,
                 log_ab=None):

        self.vae = vae
        self.K = self.likelihood.K
        self.pi = pi if pi is not None else \
            torch.ones(self.K, dtype=torch.float64, requires_grad=True)
        self.log_ab = log_ab if log_ab is not None else \
            torch.randn(self.K, 2, dtype=torch.float64, requires_grad=True)
        self.model_params = [self.pi, self.log_ab]

    def log_like(self, Y, dT):

        prob = []
        self.pi_norm = torch.nn.LogSoftmax(dim=0)(self.pi)
        for k in range(self.K):
            p_g = Gamma(torch.exp(self.log_ab[k][0]), torch.exp(self.log_ab[k][1]))
            prob_g = p_g.log_prob(dT)
            prob.append(prob_g + self.pi_norm[k])
        prob = torch.stack(prob).t() + self.likelihood.mixture_prob(Y)

        return prob

    def loss(self, Y, dT):

        q = self.vae.encode(Y)
        h = q.rsample()
        like_Y = self.vae.decode(h).log_prob(Y).sum()
        ent = q.entropy().sum()
        ll = self.log_like(h, dT)

        return -ll.logsumexp(dim=1).sum() - ent - like_Y