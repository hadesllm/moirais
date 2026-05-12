# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian shrinkage (horseshoe prior)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_horseshoe(
    X: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    *,
    n_iter: int = 3000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bayesian linear regression with horseshoe prior via Gibbs sampling.

    Prior: beta_j | lambda_j, tau ~ N(0, lambda_j^2 * tau^2)
           lambda_j ~ C+(0, 1), tau ~ C+(0, 1)

    :param X: Design matrix (n, p).
    :param y: Response vector (n,).
    :param n_iter: Number of Gibbs iterations.
    :param seed: Random seed.
    :return: Dictionary with beta_samples, tau_samples, posterior_mean.

    References
    ----------
    Carvalho, C. M., et al. (2010). *Biometrika*, 97(2), 465--480.
    """
    rng = np.random.default_rng(seed)
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, p = X_arr.shape

    beta = np.zeros(p)
    sigma2 = 1.0
    tau2 = 1.0
    lambda2 = np.ones(p)
    nu = np.ones(p)
    xi = 1.0

    beta_samples = np.empty((n_iter, p))
    tau_samples = np.empty(n_iter)

    XtX = X_arr.T @ X_arr
    Xty = X_arr.T @ y_arr

    for it in range(n_iter):
        D_inv = np.diag(1.0 / (lambda2 * tau2 + 1e-30))
        prec = XtX / sigma2 + D_inv
        cov = np.linalg.inv(prec)
        mean = cov @ (Xty / sigma2)
        beta = rng.multivariate_normal(mean, cov)

        resid = y_arr - X_arr @ beta
        post_a = (n + p) / 2.0 + 1
        post_b = 0.5 * float(resid @ resid) + 0.5 * float(np.sum(beta ** 2 / (lambda2 * tau2 + 1e-30)))
        sigma2 = 1.0 / rng.gamma(post_a, 1.0 / post_b)

        for j in range(p):
            rate = beta[j] ** 2 / (2 * sigma2 * tau2 + 1e-30) + 1.0 / (nu[j] + 1e-30)
            lambda2[j] = 1.0 / rng.gamma(1.0, 1.0 / (rate + 1e-30))

        rate_tau = np.sum(beta ** 2 / (lambda2 * sigma2 + 1e-30)) / 2.0 + 1.0 / (xi + 1e-30)
        tau2 = 1.0 / rng.gamma((p + 1) / 2.0, 1.0 / (rate_tau + 1e-30))

        for j in range(p):
            nu[j] = 1.0 / rng.gamma(1.0, 1.0 / (1.0 + 1.0 / (lambda2[j] + 1e-30)))
        xi = 1.0 / rng.gamma(1.0, 1.0 / (1.0 + 1.0 / (tau2 + 1e-30)))

        beta_samples[it] = beta
        tau_samples[it] = np.sqrt(tau2)

    return {
        "beta_samples": beta_samples,
        "tau_samples": tau_samples,
        "posterior_mean": np.mean(beta_samples, axis=0).tolist(),
        "posterior_sd": np.std(beta_samples, axis=0, ddof=1).tolist(),
        "n_iter": n_iter,
    }


bshrk = bayesian_horseshoe


def cheatsheet() -> str:
    return "bayesian_horseshoe({}) -> Bayesian shrinkage (horseshoe prior)."
