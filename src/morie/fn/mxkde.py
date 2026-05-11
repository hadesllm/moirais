# morie.fn — function file (hadesllm/morie)
"""Mixture model KDE."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mxkde(
    x: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    n_components: int = 3,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_grid: int = 256,
    seed: int | None = None,
) -> dict:
    r"""
    Gaussian mixture model density estimation via EM algorithm.

    Fits :math:`f(x) = \sum_{k=1}^{K} \pi_k \phi(x; \mu_k, \sigma_k^2)`
    where :math:`\phi` is the normal density.

    Parameters
    ----------
    x : np.ndarray
        Data vector (n,).
    x_eval : np.ndarray or None
        Evaluation grid.
    n_components : int
        Number of mixture components K.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood.
    n_grid : int
        Grid size when ``x_eval`` is None.
    seed : int or None
        RNG seed for initial means.

    Returns
    -------
    dict
        ``x_eval``, ``density``, ``weights``, ``means``, ``stds``,
        ``log_likelihood``, ``n_iter``, ``n_obs``.

    References
    ----------
    McLachlan, G. & Peel, D. (2000). Finite Mixture Models. Wiley.
    Horowitz (2009). Ch 2.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 observations.")
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    rng = np.random.default_rng(seed)
    K = n_components

    idx = np.sort(rng.choice(n, size=K, replace=False))
    mu = x[idx].copy()
    sigma = np.full(K, np.std(x, ddof=1))
    pi_k = np.full(K, 1.0 / K)

    ll_old = -np.inf
    n_iter = 0
    for iteration in range(max_iter):
        resp = np.zeros((n, K))
        for k in range(K):
            resp[:, k] = pi_k[k] * stats.norm.pdf(x, mu[k], sigma[k])
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum = np.maximum(resp_sum, 1e-300)
        resp /= resp_sum

        ll = np.sum(np.log(np.maximum(resp.sum(axis=1) * resp_sum.ravel() / resp_sum.ravel(), 1e-300)))
        ll = np.sum(np.log(np.maximum(
            sum(pi_k[k] * stats.norm.pdf(x, mu[k], sigma[k]) for k in range(K)),
            1e-300,
        )))

        n_iter = iteration + 1
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll

        Nk = resp.sum(axis=0)
        for k in range(K):
            if Nk[k] < 1e-10:
                continue
            pi_k[k] = Nk[k] / n
            mu[k] = (resp[:, k] * x).sum() / Nk[k]
            sigma[k] = np.sqrt((resp[:, k] * (x - mu[k]) ** 2).sum() / Nk[k] + 1e-10)

    if x_eval is None:
        lo = x.min() - 3 * np.max(sigma)
        hi = x.max() + 3 * np.max(sigma)
        x_eval = np.linspace(lo, hi, n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    density = np.zeros(len(x_eval))
    for k in range(K):
        density += pi_k[k] * stats.norm.pdf(x_eval, mu[k], sigma[k])

    return {
        "x_eval": x_eval.tolist(),
        "density": density.tolist(),
        "weights": pi_k.tolist(),
        "means": mu.tolist(),
        "stds": sigma.tolist(),
        "log_likelihood": float(ll_old),
        "n_iter": n_iter,
        "n_obs": n,
    }


mxkde_fn = mxkde


def cheatsheet() -> str:
    return "mxkde({x}) -> Gaussian mixture model KDE via EM."
