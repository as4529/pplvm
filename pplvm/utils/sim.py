import numpy as np
from .utils import softmax, softplus
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

def sim_rand_sinusoid(weights_trans, weights_time,
                      bias_trans,
                      bias_time,
                      obs_means,
                      N=1000,
                      K=3, C=2,
                      p_max=2,
                      h_max=1.,
                      y_var=1.,
                      plot=False):

    H_sin = np.random.uniform(-1, 1, size=(1, C)) * h_max
    H_cos = np.random.uniform(-1, 1, size=(1, C)) * h_max
    P_sin = np.random.uniform(size=(1, C)) * p_max
    P_cos = np.random.uniform(size=(1, C)) * p_max
    z = np.random.choice(K)
    D = obs_means.shape[1]
    X, Z, Y, dT, pi, alpha_beta = ([] for i in range(6))
    t = 0

    for n in range(N):
        x_t = np.sum(np.sin(t * np.pi / N * P_sin) * H_sin, axis=1) + \
              np.sum(np.cos(t * np.pi / N * P_cos) * H_cos, axis=1)
        pi_t = bias_trans + weights_trans * x_t
        pi_exp = np.exp(pi_t)
        pi_t = pi_exp / np.expand_dims(pi_exp.sum(axis=1), 1)

        z = np.random.choice(K, p=pi_t[z, :])
        y = np.random.normal(0, y_var, size=D) + obs_means[z]

        ab_n = softplus(weights_time * x_t + bias_time)
        dt = np.random.gamma(ab_n[z][0], 1. / ab_n[z][1])
        t += dt

        X.append(x_t)
        Z.append(z)
        Y.append(y)
        dT.append(dt)
        alpha_beta.append(ab_n)
        pi.append(pi_t)

    X = np.array(X).squeeze()
    Z = np.array(Z).squeeze()
    Y = np.array(Y).squeeze()
    dT = np.array(dT).squeeze()

    X_all = []
    pi_all = []
    for t in range(int(np.sum(dT))):
        x_t = np.sum(np.sin(t * np.pi / N * P_sin) * H_sin, axis=1) + \
              np.sum(np.cos(t * np.pi / N * P_cos) * H_cos, axis=1)
        pi_t = bias_trans + weights_trans * x_t
        pi_exp = np.exp(pi_t)
        pi_t = pi_exp / np.expand_dims(pi_exp.sum(axis=1), 1)
        X_all.append(x_t)
        pi_all.append(pi_t)
    X_all = np.array(X_all).squeeze()
    pi_all = np.stack(pi_all)

    if plot:
        plot_sim(dT, X_all, pi_all, Y, Z, K)

    return X, Y, Z, dT, np.stack(pi), np.stack(alpha_beta),\
           H_sin, H_cos, P_sin, P_cos, X_all, pi_all

def plot_sim(dT, X_all, pi_all, Y, Z, K):

    T = np.cumsum(dT)
    iplot([go.Scatter(x=list(range(int(np.sum(dT)))), y=X_all)])
    for i in range(K):
        pi0_pred = pi_all[:, i, :]
        iplot([go.Scatter(x=list(range(int(T[-1]))), y=pi0_pred[:, j],
                          line=dict(shape='spline')) for j in range(K)])
    iplot([go.Scatter(x=T, y=Y[:,0], mode='markers', marker=dict(color=Z))])
    return
