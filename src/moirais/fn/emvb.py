# moirais.fn — function file (hadesllm/moirais)
"""Expectation-Maximization Variational Bayes (Gaussian mixture)."""

from __future__ import annotations

__all__ = ["em_variational_bayes", "emvb"]

from typing import Any, Union

import numpy as np
from scipy.special import digamma


def em_variational_bayes(
    data: Union[list, np.ndarray],
    n_components: int = 2,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Variational Bayes EM for a univariate Gaussian mixture model.

    Uses conjugate priors (Normal-Gamma for component parameters,
    Dirichlet for mixing weights) and optimizes the evidence lower
    bound (ELBO) via coordinate ascent variational inference.

    Parameters
    ----------
    data : array-like
        Observed data points (n,).
    n_components : int
        Number of mixture components K.
    max_iter : int
        Maximum VB-EM iterations.
    tol : float
        ELBO convergence tolerance.
    seed : int
        Random seed for initialization.

    Returns
    -------
    dict
        means : ndarray (K,)  -- posterior mean of each component mean
        precisions : ndarray (K,)  -- posterior mean of each precision
        weights : ndarray (K,)  -- expected mixing proportions
        responsibilities : ndarray (n, K)
        elbo_history : list of float
        converged : bool

    References
    ----------
    Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*,
    Springer, Ch. 10.
    Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017).
    Variational inference: A review for statisticians.
    *JASA*, 112(518), 859--877.
    """
    x = np.asarray(data, dtype=float).ravel()
    n = len(x)
    K = n_components

    if n < K:
        raise ValueError("Need at least as many data points as components.")

    rng = np.random.default_rng(seed)

    alpha_0 = 1.0
    m_0 = float(np.mean(x))
    beta_0 = 0.01
    a_0 = 0.5
    b_0 = 0.5

    alpha = np.full(K, alpha_0 + n / K)
    quantiles = np.linspace(0, 1, K + 2)[1:-1]
    m = np.quantile(x, quantiles)
    beta = np.full(K, beta_0 + n / K)
    a = np.full(K, a_0 + n / (2.0 * K))
    b = np.full(K, b_0)

    elbo_history = []
    converged = False

    for iteration in range(max_iter):
        E_ln_pi = digamma(alpha) - digamma(np.sum(alpha))
        E_ln_tau = digamma(a) - np.log(b)
        E_tau = a / b

        log_rho = np.zeros((n, K))
        for k in range(K):
            diff = x - m[k]
            log_rho[:, k] = (
                E_ln_pi[k]
                + 0.5 * E_ln_tau[k]
                - 0.5 * E_tau[k] * (diff ** 2 + 1.0 / beta[k])
            )

        log_rho -= np.max(log_rho, axis=1, keepdims=True)
        r = np.exp(log_rho)
        r /= np.sum(r, axis=1, keepdims=True)

        N_k = np.sum(r, axis=0) + 1e-30
        x_bar_k = r.T @ x / N_k
        S_k = np.zeros(K)
        for k in range(K):
            S_k[k] = np.sum(r[:, k] * (x - x_bar_k[k]) ** 2) / N_k[k]

        alpha = alpha_0 + N_k
        beta = beta_0 + N_k
        m = (beta_0 * m_0 + N_k * x_bar_k) / beta
        a = a_0 + 0.5 * N_k
        b = b_0 + 0.5 * N_k * (S_k + (beta_0 * N_k * (x_bar_k - m_0) ** 2) / beta)

        elbo = float(np.sum(r * log_rho) - np.sum(r * np.log(r + 1e-30)))
        elbo_history.append(elbo)

        if iteration > 0 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            converged = True
            break

    weights = alpha / np.sum(alpha)

    return {
        "means": m,
        "precisions": a / b,
        "weights": weights,
        "responsibilities": r,
        "elbo_history": elbo_history,
        "converged": converged,
        "n_components": K,
        "n_iter": iteration + 1,
    }


emvb = em_variational_bayes


def cheatsheet() -> str:
    return "em_variational_bayes(data, n_components) -> VB-EM for Gaussian mixture."
