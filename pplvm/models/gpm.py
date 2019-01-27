import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal, Gamma, kl_divergence

from pplvm.message_passing.normalizer import hmmnorm_cython

class GPM(nn.Module):
    """
    Gaussian process modulated Markov renewal process
    """
    def __init__(self,
                 K,
                 obs_model,
                 ls,
                 var,
                 induce_rate=50,
                 x_model=None,
                 A=None,
                 log_ab=None):
        """
        
        Args:
            K (int): number of discrete states 
            obs_model (nn.Module): see pplvm.models.likelihoods for examples 
            ls (float): GP prior lengthscale 
            var (float): GP prior variance 
            induce_rate (int): "inducing rate" defined as number of points between
                                "inducing" points 
            x_model (nn.Module): recognition network that maps from embeddings (or 
                                 observed data) to belief of continuous states
                                 at inducing points (see pplvm.models.rnn) 
            A (torch.tensor): initial base transition matrix (K by K) 
            log_ab (torch.tensor): initial Gamma parameters (K by 2)
             
        """
        super(GPM, self).__init__()
        self.K = K
        self.obs_model = obs_model

        self.ls = ls
        self.var = var
        self.induce_rate = induce_rate

        self.x_model = x_model

        self.bias_A = A if A is not None else \
            nn.Parameter(torch.ones(self.K, self.K, dtype=torch.float64))
        self.bias_ab = log_ab if log_ab is not None else \
            nn.Parameter(torch.randn(self.K, 2, dtype=torch.float64))

        self.W_pi = nn.Parameter(torch.randn(self.K, 1, dtype=torch.float64))
        self.W_ab = nn.Parameter(torch.randn(self.K, 2, dtype=torch.float64))

    def forward(self,
                Y,
                dT,
                T,
                K_uu_inv=None,
                K_xu=None,
                qmu=None,
                qs=None,
                anneal=1.):
        """
        
        Args:
            Y (torch.tensor): observed marks 
            dT (torch.tensor): observed intervals 
            T (torch.tensor): observed timestamps (in case there are gaps that aren't
                              accounted for in dT) 
            K_uu_inv (torch.tensor): n_induce by n_induce inverse covariance matrix
                                     at inducing points
            K_xu (torch.tensor): N by n_induce covariance matrix between observed data
                                 and inducing points
            qmu (variational mean): use this if you are not using amortization
            qs (variational log variance): use this if you are not amortizing
            anneal (float): factor to scale KL loss

        Returns:
            sampled continuous states (X), ELBO loss

        """

        # Calculating covariance
        T_induce, induce_idx = self.get_T_induce(T)
        if K_uu_inv is None:
            K_uu_inv = self.calc_K_inv(T_induce)
        if K_xu is None:
            K_xu = self.calc_K_xu(T_induce, T)

        # getting embedding (if using VAE) and observation loss
        h, y_ll, y_loss = self.obs_model(Y)

        # GP loss
        if qmu is not None and qs is not None:
            q_u = MultivariateNormal(qmu, torch.diag(qs.exp()))
        else:
            mu, log_var = self.x_model(h, dT, induce_idx, switch)
            q_u = MultivariateNormal(mu.squeeze(),
                                     torch.diag(log_var.squeeze().exp()))
        # sparse GP KL
        u = q_u.rsample().squeeze()
        p_u = MultivariateNormal(torch.zeros(u.shape[0], dtype=torch.float64),
                                 precision_matrix=K_uu_inv)
        kl = kl_divergence(q_u, p_u)

        # HMM loss
        X = torch.mv(K_xu, torch.mv(K_uu_inv, u))
        log_pi0, log_pi, log_ab = self.calc_params(X)
        ll = self.log_like_dT(dT, log_ab) + y_ll

        loss_hmm = -1. * hmmnorm_cython(log_pi0, log_pi.contiguous(),  ll.contiguous()) +\
                    y_loss
        loss = loss_hmm + anneal * kl

        return X, loss

    def calc_params(self, X):
        """
        Calculates parameters of non-stationary transitions and interval distributions
        Args:
            X (torch.tensor): N by D_x tensor of continuous states

        Returns:
            initial state probabilities, N by K by K transition matrices,
            N by K by 2 Gamma interval parameters
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
            T_induce (torch.tesnsor)): time of inducing points
            eps (float): jitter to add to diagonal before inverting

        Returns: inverse of covariance matrix evaluated at inducing points
                (n_induce, n_induce)

        """

        dist = torch.pow(T_induce.unsqueeze(1) - T_induce, 2)
        K = self.var * torch.exp(- dist / (2 * self.ls ** 2))
        K = K + torch.eye(K.shape[0]).double() * eps
        K_inv = K.inverse()

        return K_inv

    def calc_K_xu(self, T_induce, T):
        """
        Calculates covariance between inducing points
        Args:
            T_induce (torch.tensor): times of inducing points
            T (torch.tensor): times of all points

        Returns: (n, n_induce) covariance matrix

        """

        dist = torch.pow(T_induce - T.unsqueeze(1), 2)
        W = self.var * torch.exp(- dist / (2 * self.ls ** 2))

        return W

    def get_T_induce(self, T):
        """
        calculates inducing points based on inducing rate and timestamps
        Args:
            T (torch.tensor): timestamps

        Returns: inducing points (n_induce)

        """

        induce_idx = list(np.arange(0, len(T) - 1, self.induce_rate).astype(int))
        T_induce = T[induce_idx]

        return T_induce, induce_idx

    def log_like_dT(self, dT, log_ab):
        """
        Calculates per-class log likelihood of latent embeding and intervals
        Args:
            dT (torch.tensor): interbout intervals (T)
            log_ab (torch.tensor): per-class parameters (K, 2, N)

        Returns: (N, K)

        """

        prob = []
        for k in range(self.K):
            p_g = Gamma(torch.exp(log_ab[k][0]), torch.exp(log_ab[k][1]))
            prob_g = p_g.log_prob(dT)
            prob.append(prob_g)
        prob = torch.stack(prob).t()

        return prob

    def predict(self, Y, dT, T,  qmu=None, qs=None):
        """
        Makes a prediction of latent states given observed data, intervals,
        and timestamps
        Args:
            Y (torch.tensor): observed marks
            dT (torch.tensor): observed intervals
            T (torch.tensor): observed timestamps (in case there are gaps that aren't
                              accounted for in dT)
            qmu (variational mean): use this if you are not using amortization
            qs (variational log variance): use this if you are not amortizing

        """

        T_induce, induce_idx = self.get_T_induce(T)
        K_uu_inv = self.calc_K_inv(T_induce)
        K_xu = self.calc_K_xu(T_induce, T)

        h, y_ll, y_loss = self.obs_model(Y)

        if qmu is not None and qs is not None:
            q_u = Normal(qmu, qs.exp())
        else:
            mu, log_var = self.x_model(h, dT, induce_idx)
            q_u = MultivariateNormal(mu.squeeze(),
                                     torch.diag(log_var.squeeze().exp()))
        u = q_u.mean.squeeze()
        X = torch.mv(K_xu, torch.mv(K_uu_inv, u))

        log_pi0, log_pi, log_ab = self.calc_params(X)
        ll = self.log_like_dT(dT, log_ab) + y_ll
        T, K = ll.shape
        delta = log_pi0 + ll[0]
        idx = []

        # forward
        for t in range(1, T):
            max_idx = torch.max(delta + log_pi[t - 1].t(), dim=1)
            idx.append(max_idx[1])
            delta = max_idx[0] + ll[t]
        i = torch.argmax(delta)
        Z = [i]

        # backward
        for t in range(T - 2, -1, -1):
            i = idx[t][i]
            Z.append(i)

        pi = log_pi.exp()
        ab = log_ab.exp()

        return list(reversed(Z)), X, pi, ab