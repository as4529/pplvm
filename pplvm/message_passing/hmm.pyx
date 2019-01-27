# Cython implementation of message passing
#
# distutils: extra_compile_args = -O3
# cython: wraparound=True
# cython: boundscheck=True
# cython: nonecheck=True
# cython: cdivision=True

import numpy as np
cimport numpy as np

from libc.math cimport log, exp, fmax

cdef double logsumexp(double[::1] x):
    cdef int i, N
    cdef double m, out

    N = x.shape[0]

    # find the max
    m = -np.inf
    for i in range(N):
        m = fmax(m, x[i])

    # sum the exponentials
    out = 0
    for i in range(N):
        out += exp(x[i] - m)

    return m + log(out)


cdef dlse(double[::1] a,
          double[::1] out):

    cdef int K, k
    K = a.shape[0]
    cdef double lse = logsumexp(a)

    for k in range(K):
        out[k] = exp(a[k] - lse)

cpdef forward_pass(double[:,::1] log_pi0,
                   double[:,:,:,::1] log_As,
                   double[:,:,::1] log_likes,
                   double[:,:,::1] alphas):

    cdef int B, T, K, t, k
    B = log_likes.shape[0]
    T = log_likes.shape[1]
    K = log_likes.shape[2]
    assert log_As.shape[1] == T-1
    assert log_As.shape[2] == K
    assert log_As.shape[3] == K
    assert alphas.shape[1] == T
    assert alphas.shape[2] == K
    cdef double[:,::1] tmp = np.zeros((B, K))
    cdef double[::1] out = np.zeros(B)

    for b in range(B):

        for k in range(K):
            alphas[b, 0, k] = log_pi0[b, k] + log_likes[b, 0, k]

        for t in range(T - 1):
            for k in range(K):
                for j in range(K):
                    tmp[b, j] = alphas[b, t, j] + log_As[b, t, j, k]
                alphas[b, t+1, k] = logsumexp(tmp[b,:]) + log_likes[b, t+1, k]
        out[b] = logsumexp(alphas[b, T-1])

    return out


cpdef backward_pass(double[:, :,:,::1] log_As,
                    double[:, :,::1] alphas,
                    double[:, ::1] d_log_pi0,
                    double[:, :,:,::1] d_log_As,
                    double[:, :,::1] d_log_likes):

    cdef int B, T, K, t, k, j

    B = alphas.shape[0]
    T = alphas.shape[1]
    K = alphas.shape[2]
    assert log_As.shape[1] == d_log_As.shape[1] == T-1
    assert log_As.shape[2] == d_log_As.shape[2] == K
    assert log_As.shape[3] == d_log_As.shape[3] == K
    assert d_log_pi0.shape[1] == K
    assert d_log_likes.shape[1] == T
    assert d_log_likes.shape[2] == K

    # Initialize temp storage for gradients
    cdef double[:,::1] tmp1 = np.zeros((B, K))
    cdef double[:,:, ::1] tmp2 = np.zeros((B, K, K))


    for b in range(B):

        dlse(alphas[b,T-1], d_log_likes[b,T-1])
        for t in range(T-1, 0, -1):
            for k in range(K):
                for j in range(K):
                    tmp1[b,j] = alphas[b,t-1, j] + log_As[b,t-1, j, k]
                dlse(tmp1[b], tmp2[b,k])

            for j in range(K):
                for k in range(K):
                    d_log_As[b,t-1, j, k] = d_log_likes[b,t, k] * tmp2[b,k, j]

            for k in range(K):
                d_log_likes[b,t-1, k] = 0
                for j in range(K):
                    d_log_likes[b,t-1, k] += d_log_likes[b,t, j] * tmp2[b,j, k]

        for k in range(K):
            d_log_pi0[b,k] = d_log_likes[b,0, k]


