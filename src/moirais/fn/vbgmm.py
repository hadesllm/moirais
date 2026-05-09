"""Variational Bayes Gaussian mixture model."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.special import digamma


def vb_gaussian_mixture(
    X: Union[list, np.ndarray],
    *,
    K: int = 3,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Variational Bayes inference for a Gaussian mixture model.

    Fits a K-component diagonal-covariance GMM using coordinate-ascent VI.

    :param X: Data matrix (n, d).
    :param K: Number of mixture components.
    :param max_iter: Maximum VI iterations.
    :param tol: ELBO convergence tolerance.
    :param seed: Random seed.
    :return: Dictionary with responsibilities, means, precisions, weights, elbo.

    References
    ----------
    Bishop, C. (2006). *Pattern Recognition and Machine Learning*, Ch. 10.2.
    """
    rng = np.random.default_rng(seed)
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, d = X_arr.shape

    alpha_0 = 1.0 / K
    beta_0 = 1.0
    m_0 = np.mean(X_arr, axis=0)
    W_0_inv = np.var(X_arr, axis=0) + 1e-6
    nu_0 = float(d)

    alpha = np.full(K, alpha_0 + n / K)
    beta = np.full(K, beta_0 + n / K)
    m = X_arr[rng.choice(n, K, replace=False)].copy()
    W_inv = np.tile(W_0_inv, (K, 1))
    nu = np.full(K, nu_0 + n / K)

    r = np.zeros((n, K))
    prev_elbo = -np.inf

    for it in range(max_iter):
        E_ln_pi = digamma(alpha) - digamma(np.sum(alpha))
        E_ln_prec = np.zeros(K)
        for k in range(K):
            E_ln_prec[k] = np.sum(digamma(0.5 * (nu[k] + 1 - np.arange(1, d + 1)))) + d * np.log(2) - np.sum(np.log(W_inv[k] + 1e-30))

        for k in range(K):
            diff = X_arr - m[k]
            mahal = np.sum(diff ** 2 * nu[k] / (W_inv[k] + 1e-30), axis=1)
            r[:, k] = E_ln_pi[k] + 0.5 * E_ln_prec[k] - 0.5 * d / beta[k] - 0.5 * mahal

        r_max = np.max(r, axis=1, keepdims=True)
        log_r = r - r_max
        r = np.exp(log_r)
        r = r / (np.sum(r, axis=1, keepdims=True) + 1e-30)

        N_k = np.sum(r, axis=0) + 1e-10
        x_bar = (r.T @ X_arr) / N_k[:, None]

        for k in range(K):
            alpha[k] = alpha_0 + N_k[k]
            beta[k] = beta_0 + N_k[k]
            m[k] = (beta_0 * m_0 + N_k[k] * x_bar[k]) / beta[k]
            diff = X_arr - x_bar[k]
            S_k = np.sum(r[:, k:k+1] * diff ** 2, axis=0) / N_k[k]
            dm = x_bar[k] - m_0
            W_inv[k] = W_0_inv + N_k[k] * S_k + (beta_0 * N_k[k] / beta[k]) * dm ** 2
            nu[k] = nu_0 + N_k[k]

        elbo = float(np.sum(r * (E_ln_pi[None, :] - np.log(r + 1e-30))))
        if abs(elbo - prev_elbo) < tol:
            break
        prev_elbo = elbo

    weights = alpha / np.sum(alpha)

    return {
        "responsibilities": r,
        "means": m.tolist(),
        "weights": weights.tolist(),
        "alpha": alpha.tolist(),
        "elbo": float(elbo),
        "n_iter": it + 1,
        "K": K,
    }


vbgmm = vb_gaussian_mixture


def cheatsheet() -> str:
    return "vb_gaussian_mixture({}) -> Variational Bayes Gaussian mixture model."
