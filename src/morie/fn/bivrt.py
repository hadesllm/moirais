# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian IV regression."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_iv(
    y: Union[list, np.ndarray],
    X: Union[list, np.ndarray],
    Z: Union[list, np.ndarray],
    *,
    n_iter: int = 5000,
    seed: int = 42,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Bayesian instrumental variable regression via Gibbs sampling.

    Two-stage model with conjugate priors.

    :param y: Outcome variable (n,).
    :param X: Endogenous regressor(s) (n,) or (n, p).
    :param Z: Instruments (n,) or (n, q).
    :param n_iter: Number of Gibbs iterations.
    :param seed: Random seed.
    :param prob: Credible interval probability.
    :return: Dictionary with beta_samples, posterior mean, CIs.

    References
    ----------
    Kleibergen, F. & Zivot, E. (2003). *J. Econometrics*, 114, 29--72.
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=float).ravel()
    X_arr = np.asarray(X, dtype=float)
    Z_arr = np.asarray(Z, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    n, p = X_arr.shape

    Pz = Z_arr @ np.linalg.lstsq(Z_arr, X_arr, rcond=None)[0]
    XtPx = Pz.T @ Pz
    Pty = Pz.T @ y_arr

    beta = np.zeros(p)
    sigma2 = 1.0

    beta_samples = np.empty((n_iter, p))
    alpha_prior = 0.01

    for it in range(n_iter):
        prec = XtPx / sigma2 + alpha_prior * np.eye(p)
        cov = np.linalg.inv(prec)
        mean = cov @ (Pty / sigma2)
        beta = rng.multivariate_normal(mean, cov)

        resid = y_arr - X_arr @ beta
        post_a = (n + 1) / 2.0
        post_b = 0.5 * float(resid @ resid) + 0.5
        sigma2 = 1.0 / rng.gamma(post_a, 1.0 / post_b)

        beta_samples[it] = beta

    alpha_half = (1 - prob) / 2
    ci_lo = np.percentile(beta_samples, 100 * alpha_half, axis=0)
    ci_hi = np.percentile(beta_samples, 100 * (1 - alpha_half), axis=0)

    return {
        "beta_samples": beta_samples,
        "posterior_mean": np.mean(beta_samples, axis=0).tolist(),
        "posterior_sd": np.std(beta_samples, axis=0, ddof=1).tolist(),
        "ci_lower": ci_lo.tolist(),
        "ci_upper": ci_hi.tolist(),
        "n": n,
    }


bivrt = bayesian_iv


def cheatsheet() -> str:
    return "bayesian_iv({}) -> Bayesian IV regression."
