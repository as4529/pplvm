import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal, Gamma, kl_divergence

from pplvm.message_passing.normalizer import hmmnorm_cython
from pplvm.models.gpm import GPM


class GPMVAE(GPM):

    def __init__(self,
                 vae,
                 ls,
                 var,
                 induce_rate=20,
                 x_model=None,
                 A=None,
                 log_ab=None):
        self.vae = vae
        self.K = self.vae.K
        super(GPMVAE, self).__init__(likelihood=None,
                                     ls=ls,
                                     var=var,
                                     induce_rate=induce_rate,
                                     x_model=x_model,
                                     A=A,
                                     log_ab=log_ab)
        self.pi = self.vae.pi

    def loss(self, Y, dT, T, switch=None,
             K_zz_inv=None, K_xz=None,
             qmu=None, qs=None, anneal=1., eps=1e-3):

        # VAE loss
        q_h = self.vae.encode(Y)
        ent_h = q_h.entropy().sum()
        h = q_h.rsample()
        like_y = self.vae.decode(h).log_prob(Y).sum()
        loss_h = - like_y - ent_h

        # Calculating covariance
        T_induce, induce_idx = self.get_T_induce(T)
        if K_zz_inv is None:
            K_zz_inv = self.calc_K_inv(T_induce)
        if K_xz is None:
            K_xz = self.calc_K_xz(T_induce, T)

        # GP loss
        if qmu is not None and qs is not None:
            q_u = MultivariateNormal(qmu, torch.diag(qs.exp() + eps))
        else:
            mu, log_var = self.x_model(h, dT, induce_idx, switch)
            q_u = MultivariateNormal(mu.squeeze(), torch.diag(log_var.squeeze().exp() + eps))

        # sparse GP KL
        u = q_u.rsample().squeeze()
        p_u = MultivariateNormal(torch.zeros(u.shape[0], dtype=torch.float64),
                                 precision_matrix=K_zz_inv)
        kl = kl_divergence(q_u, p_u)

        # HMM loss
        X = torch.mv(K_xz, torch.mv(K_zz_inv, u))
        log_pi0, log_pi, log_ab = self.calc_params(X)
        ll = self.log_like_dT(dT, log_ab) + self.vae.mixture_prob(h)
        loss_hmm = -1. * hmmnorm_cython(log_pi0, log_pi.contiguous(),  ll.contiguous())

        loss = loss_hmm + anneal * kl + loss_h

        return loss

    def predict(self, Y, dT, T, switch=None, qmu=None, qs=None):

        T_induce, induce_idx = self.get_T_induce(T)
        K_inv = self.calc_K_inv(T_induce)
        W = self.calc_K_xz(T_induce, T)
        h = self.vae.encode(Y).mean

        if qmu is not None and qs is not None:
            q_u = Normal(qmu, qs.exp())
        else:
            mu, log_var = self.x_model(h, dT, induce_idx, switch)
            q_u = MultivariateNormal(mu.squeeze(), torch.diag(log_var.squeeze().exp()))
        u = q_u.mean.squeeze()
        X = torch.mv(W, torch.mv(K_inv, u))

        log_pi0, log_pi, log_ab = self.calc_params(X)
        ll = self.log_like_dT(dT, log_ab) + self.vae.mixture_prob(h)
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