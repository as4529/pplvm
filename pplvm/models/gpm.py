import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal, Gamma, kl_divergence

from pplvm.message_passing.normalizer import hmmnorm_cython

class GPM(object):

    def __init__(self,
                 likelihood,
                 ls,
                 var,
                 induce_rate=50,
                 x_model=None,
                 A=None,
                 log_ab=None):

        self.likelihood = likelihood
        if self.likelihood is not None:
            self.K = self.likelihood.K

        self.ls = ls
        self.var = var
        self.induce_rate = induce_rate

        self.x_model = x_model

        self.pi = torch.ones(self.K, dtype=torch.float64, requires_grad=True)
        self.bias_A = A if A is not None else \
            torch.ones(self.K, self.K, dtype=torch.float64, requires_grad=True)
        self.bias_ab = log_ab if log_ab is not None else \
            torch.randn(self.K, 2, dtype=torch.float64, requires_grad=True)

        self.W_pi = torch.randn(self.K, 1, dtype=torch.float64, requires_grad=True)
        self.W_ab = torch.randn(self.K, 2, dtype=torch.float64, requires_grad=True)

    def loss(self, Y, dT, T,
             switch=None, K_zz_inv=None,
             K_xz=None, qmu=None, qs=None, anneal=1.):

        # Calculating covariance
        T_induce, induce_idx = self.get_T_induce(T)
        if K_zz_inv is None:
            K_zz_inv = self.calc_K_inv(T_induce)
        if K_xz is None:
            K_xz = self.calc_K_xz(T_induce, T)

        # GP loss
        if qmu is not None and qs is not None:
            q_u = MultivariateNormal(qmu, torch.diag(qs.exp()))
        else:
            mu, log_var = self.x_model(Y, dT, induce_idx, switch)
            q_u = MultivariateNormal(mu.squeeze(),
                                     torch.diag(log_var.squeeze().exp()))
        # sparse GP KL
        u = q_u.rsample().squeeze()
        p_u = MultivariateNormal(torch.zeros(u.shape[0], dtype=torch.float64),
                                 precision_matrix=K_zz_inv)
        kl = kl_divergence(q_u, p_u)

        # HMM loss
        X = torch.mv(K_xz, torch.mv(K_zz_inv, u))
        log_pi0, log_pi, log_ab = self.calc_params(X)
        ll = self.log_like_dT(dT, log_ab) + self.likelihood.mixture_prob(Y)
        loss_hmm = -1. * hmmnorm_cython(log_pi0, log_pi.contiguous(),  ll.contiguous())

        loss = loss_hmm + anneal * kl

        return loss

    def calc_params(self, X):
        """
        Returns parameters for non-stationary HMM and interval observations
        :param X:
        :type X:
        :return:
            log_pi0: log of initial state distribution (K)
            log_pi: log of transition matrices (N, K, K)
            log of gamma parameters: (K, 2, N)
        :rtype:
        """

        log_pi0 = torch.nn.LogSoftmax(dim=0)(self.pi)

        pi_trans = X[:-1] * self.W_pi
        pi = pi_trans.t().unsqueeze(1) + self.bias_A
        log_pi = torch.nn.LogSoftmax(dim=2)(pi)
        log_ab = X * self.W_ab.unsqueeze(2) + self.bias_ab.unsqueeze(2)

        return log_pi0, log_pi, log_ab

    def calc_K_inv(self, T_induce, eps=1e-3):
        """
        Calculates inverse covariance matrix
        Args:
            T_induce (): time of inducing points
            eps (): noise to add to diagonal before inverting

        Returns: inverse of covariance matrix evaluated at inducing points
                (n_induce, n_induce)

        """

        dist = torch.pow(T_induce.unsqueeze(1) - T_induce, 2)
        K = self.var * torch.exp(- dist / (2 * self.ls ** 2))
        K = K + torch.eye(K.shape[0]).double() * eps
        K_inv = K.inverse()

        return K_inv

    def calc_K_xz(self, T_induce, T):
        """
        Calculates covariance between inducing points
        Args:
            T_induce (): times of inducing points
            T (): times of all points

        Returns: (n, n_induce) covariance matrix

        """

        dist = torch.pow(T_induce - T.unsqueeze(1), 2)
        W = self.var * torch.exp(- dist / (2 * self.ls ** 2))

        return W

    def get_T_induce(self, T):
        """
        calculates inducing points based on inducing rate and timestamps
        Args:
            T (): timestamps

        Returns: inducing points (n_induce)

        """

        induce_idx = list(np.arange(0, len(T) - 1, self.induce_rate).astype(int))
        T_induce = T[induce_idx]

        return T_induce, induce_idx

    def log_like_dT(self, dT, log_ab):
        """
        Calculates per-class log likelihood of latent embeding and intervals
        Args:
            h (): latent embedding (N, D_h)
            dT (): interbout intervals (T)
            log_ab (): per-class parameters (K, 2, N)

        Returns: (N, K)

        """

        prob = []
        for k in range(self.K):
            p_g = Gamma(torch.exp(log_ab[k][0]), torch.exp(log_ab[k][1]))
            prob_g = p_g.log_prob(dT)
            prob.append(prob_g)
        prob = torch.stack(prob).t()

        return prob

    def predict(self, Y, dT, T, switch=None, qmu=None, qs=None):

        T_induce, induce_idx = self.get_T_induce(T)
        K_zz_inv = self.calc_K_inv(T_induce)
        K_xz = self.calc_K_xz(T_induce, T)

        if qmu is not None and qs is not None:
            q_u = Normal(qmu, qs.exp())
        else:
            mu, log_var = self.x_model(Y, dT, induce_idx, switch)
            q_u = MultivariateNormal(mu.squeeze(),
                                     torch.diag(log_var.squeeze().exp()))
        u = q_u.mean.squeeze()
        X = torch.mv(K_xz, torch.mv(K_zz_inv, u))

        log_pi0, log_pi, log_ab = self.calc_params(X)
        ll = self.log_like_dT(dT, log_ab) + self.likelihood.mixture_prob(Y)
        T, K = ll.shape
        delta = log_pi0 + ll[0]
        idx = []
        for t in range(1, T):
            max_idx = torch.max(delta + log_pi[t - 1].t(), dim=1)
            idx.append(max_idx[1])
            delta = max_idx[0] + ll[t]
        i = torch.argmax(delta)
        Z = [i]
        for t in range(T - 2, -1, -1):
            i = idx[t][i]
            Z.append(i)

        pi = log_pi.exp()
        ab = log_ab.exp()

        return list(reversed(Z)), X, pi, ab