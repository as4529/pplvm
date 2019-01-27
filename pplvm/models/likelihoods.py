import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal

class MixtureVAE(nn.Module):
    """
    VAE with a mixture prior
    """
    def __init__(self, encoder, decoder, prior):
        """

        Args:
            encoder (nn.Module): VAE encoder (x->h)
            decoder (nn.Module): VAE decoder (h->x)
            prior (nn.Module): mixture prior
        """

        super(MixtureVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.pi = self.prior.pi

    def forward(self, x):
        """

        Args:
            x (torch.tensor): input data

        Returns:
            h (torch.tensor): encoded x
            mixtureprob (torch.tensor): x.shape[0] by prior.K tensor with likelihoods
                                        by discrete state
            loss (torch.tensor): negative log likelihood minus entropy
                                 use this combined with mixtureprob to calculate elbo

        """
        q = self.encode(x)
        h = q.rsample()
        loglike = self.decode(h).log_prob(x).sum()
        mixtureprob = self.prior.mixture_prob(h)
        ent = q.entropy().sum()
        loss = - loglike - ent

        return h, mixtureprob, loss

    def encode(self, x):
        """
        Args:
            x (torch.tensor): input data
        Returns:
            torch.distributions.Normal: distribution in latent space
        """

        return Normal(*self.encoder(x))

    def decode(self, h):

        """
        Args:
            h (torch.tensor): latent tensor

        Returns:
            torch.distributions.Normal: distribution in observed space
        """

        return Normal(*self.decoder(h))


class MixtureGaussian(nn.Module):

    """
    Mixture of Gaussians prior
    """

    def __init__(self, K, D, means=None, variances=None):
        """

        Args:
            K (int): number of discrete states
            D (int): dimensionality of data
            means (torch.tensor): initial means
            variances (torch.tensor): initial variances
        """
        super(MixtureGaussian, self).__init__()

        self.K = K
        self.D = D
        if means is None:
            self.prior_mus = nn.Parameter(
                torch.tensor(torch.randn(self.K, self.D) * 1e-2,
                             dtype=torch.float64))
        else:
            self.prior_mus = means
        if variances is None:
            self.prior_sigmas = nn.Parameter(
                torch.eye(self.D, dtype=torch.float64).repeat(self.K, 1, 1))
        else:
            self.prior_sigmas = variances
        self.pi = nn.Parameter(torch.ones(self.K, dtype=torch.float64))

    def forward(self, x):
        """

        Args:
            x (torch.tensor): input data

        Returns:
            identity, mixture probability, and 0
            this is so that MixtureVAE and MixturePrior can be used interchangeably
            in the MRP and GPM modules (depending on whether you want a VAE
            observation model or a simple mixture of Gaussians)

        """

        return x, self.mixture_prob(x), 0

    def mixture_prob(self, x):
        """

        Args:
            x (torch.tensor): input data

        Returns:
            N by K torch.tensor with log probabilities for each sample by
            discrete state

        """

        prob = []
        for k in range(self.K):
            p = MultivariateNormal(self.prior_mus[k],
                                   scale_tril=self.prior_sigmas[k].tril())
            prob.append(p.log_prob(x))
        return torch.stack(prob, dim=-1)

    def prior_prob(self, x):
        """

        Args:
            x (torch.tensor): input data

        Returns:
            scalar valued loglikelihood of observing x (summed over discrete states)
        """

        self.pi_norm = torch.nn.LogSoftmax(dim=0)(self.pi)
        prob = []
        for k in range(self.K):
            p = MultivariateNormal(self.prior_mus[k],
                                   scale_tril=self.prior_sigmas[k].tril())
            prob.append(p.log_prob(x) + self.pi_norm[k])
        return torch.stack(prob, dim=-1)

    def max_z(self, x):
        """

        Args:
            x (torch.tensor): input data (N by D)

        Returns:
            length N list with most likely discrete state (int) for each sample

        """

        prob = self.prior.prior_prob(x)
        max = torch.argmax(prob, dim=1).detach().numpy()

        return max


class Encoder(nn.Module):
    """
    Sample encoder module
    """

    def __init__(self, dim_h, dim_x, dim_l=128):
        super(Encoder, self).__init__()
        self.dim_h = dim_h
        self.dim_x = dim_x
        self.dim_l = dim_l
        self.fc1 = nn.Linear(self.dim_x, self.dim_l)
        self.fc2 = nn.Linear(self.dim_l, self.dim_l)
        self.fc3 = nn.Linear(self.dim_l, self.dim_h * 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x[:, :self.dim_h], torch.exp(x[:, self.dim_h:])


class Decoder(nn.Module):
    """
    Example decoder module
    """

    def __init__(self, dim_h, dim_x, dim_l=128):
        super(Decoder, self).__init__()
        self.dim_h = dim_h
        self.dim_x = dim_x
        self.dim_l = dim_l
        self.fc1 = nn.Linear(self.dim_h, self.dim_l)
        self.fc2 = nn.Linear(self.dim_l, self.dim_l)
        self.fc3 = nn.Linear(self.dim_l, self.dim_x * 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x[:, :self.dim_x], torch.exp(x[:, self.dim_x:])