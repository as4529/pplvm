import torch
from torch.distributions import Gamma, Normal

class MRP(nn.Module):
    """
    Markov renewal process with Gamma intervals
    """

    def __init__(self,
                 obs_model,
                 K,
                 A=None,
                 log_ab=None):
        """

        Args:
            obs_model (nn.Module): likelhood model for observed data
                                   (see pplvm.models.likelihoods)
            K (int): number of discrete states
            A (torch.tensor): K by K initial transition matrix
            log_ab (torch.tensor): K by 2 initial Gamma parameters
        """

        super(MRP, self).__init__()

        self.obs_model = obs_model
        self.K = K
        self.A = A if A is not None else \
            nn.Parameter(torch.randn(self.K, self.K, dtype=torch.float64))
        self.log_ab = log_ab if log_ab is not None else \
            nn.Parameter(torch.randn(self.K, 2, dtype=torch.float64))

    def forward(self,
                Y,
                dT):
        """
        Given observed data and intervals, returns embeddings (if using VAE observation)
        loss
        Args:
            Y ():
            dT ():

        Returns:

        """

        # calculating likelihoods per discrete state for Y and intervals
        h, y_ll, y_loss = self.obs_model(Y)
        dT_ll = self.dT_mixture_prob(dT)
        ll = y_ll + dT_ll

        # normalizing transition and initial probabilities
        A_expand = torch.nn.LogSoftmax(dim=1)(self.A).expand(len(dT) - 1,
                                                             self.K, self.K)
        pi0_norm = torch.nn.LogSoftmax(dim=0)(self.obs_model.pi)

        # computing hmm normalizer
        loss = - 1. * hmmnorm_cython(pi0_norm,
                                     A_expand.contiguous(),
                                     ll.contiguous()) + y_loss

        return h, loss

    def dT_mixture_prob(self, dT):
        """
        Calculates log likelihood of intervals by discrete state
        Args:
            dT (torch.tensor): observed intervals

        Returns:
            N x K torch.tensor with log likelihood by discrete state
        """
        prob = []
        for k in range(self.K):
            p_g = Gamma(torch.exp(self.log_ab[k][0]), torch.exp(self.log_ab[k][1]))
            prob_g = p_g.log_prob(dT)
            prob.append(prob_g)
        prob = torch.stack(prob).t()

        return prob

    def max_z(self,
              Y,
              dT):
        """
        Viterbi algorithm for computing most likely sequence of discrete states
        Args:
            Y (torch.tensor): observed marks
            dT (torch.tensor): observed intervals

        Returns:
            length N list of most likely discrete states (int)
        """

        h, y_ll, y_loss = self.obs_model(Y)
        dT_ll = self.dT_mixture_prob(dT)
        ll = y_ll + dT_ll
        A_expand = torch.nn.LogSoftmax(dim=1)(self.A).expand(len(dT) - 1,
                                                             self.K, self.K)
        pi0_norm = torch.nn.LogSoftmax()(self.prior.pi)
        T, K = ll.shape
        delta = pi0_norm + ll[0]
        idx = []

        # forward
        for t in range(1, T):
            max_idx = torch.max(delta + A_expand[t - 1].t(), dim=1)
            idx.append(max_idx[1])
            delta = max_idx[0] + ll[t]
        i = torch.argmax(delta)
        Z = [i]

        # backward
        for t in range(T - 2, -1, -1):
            i = idx[t][i]
            Z.append(i)

        return list(reversed(Z))
