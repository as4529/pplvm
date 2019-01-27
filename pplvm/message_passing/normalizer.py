import torch
from torch.autograd import Function
from .hmm import forward_pass as forward_pass_cython
from .hmm import backward_pass as backward_pass_cython
import numpy as np

class HMMNormalizerCython(Function):
    """
    Accelerated HMM normalizer function
    """

    @staticmethod
    def forward(ctx, log_pi0, log_As, log_likes):
        """
        Computes HMM normalizer
        Args:
            ctx (): context
            log_pi0 (torch.tensor): log of initial state probabilities
            log_As (torch.tensor): log of transition probabilities (N by K by K)
            log_likes (torch.tensor): log likelihoods (N by K)

        Returns:

        """
        B, T, K = log_likes.shape
        log_pi0, log_As, log_likes = log_pi0.detach(),\
                                     log_As.detach(),\
                                     log_likes.detach()
        alphas = np.zeros((B, T, K))
        Z = forward_pass_cython(log_pi0.numpy(),
                                log_As.numpy(),
                                log_likes.numpy(),
                                alphas)
        ctx.save_for_backward(torch.tensor(alphas, dtype=torch.float64),
                              log_As)
        return torch.tensor(Z, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes gradients of HMM normalizer
        Args:
            ctx (): context
            grad_output (): output of forward pass

        Returns:

        """
        alphas, log_As = ctx.saved_tensors
        alphas, log_As = alphas.detach().numpy(), log_As.detach().numpy()
        B, T, K = alphas.shape

        d_log_pi0 = np.zeros((B, K))
        d_log_As = np.zeros((B, T - 1, K, K))
        d_log_likes = np.zeros((B, T, K))

        backward_pass_cython(log_As, alphas, d_log_pi0, d_log_As, d_log_likes)

        return torch.einsum('i,ij->ij',[grad_output,
                                        torch.tensor(d_log_pi0).double()]), \
               torch.einsum('i,ijkl->ijkl', [grad_output,
                                             torch.tensor(d_log_As).double()]),\
               torch.einsum('i,ijk->ijk', [grad_output,
                                           torch.tensor(d_log_likes).double()])

hmmnorm_cython = HMMNormalizerCython.apply