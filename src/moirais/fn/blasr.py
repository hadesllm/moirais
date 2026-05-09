# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian LASSO regression."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_lasso(
    X: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    *,
    lam: float = 1.0,
    n_iter: int = 3000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bayesian LASSO via Gibbs sampling with exponential-normal scale mixture.

    Prior: beta_j | tau_j^2 ~ N(0, sigma^2 * tau_j^2)
           tau_j^2 ~ Exp(lambda^2 / 2)

    :param X: Design matrix (n, p).
    :param y: Response vector (n,).
    :param lam: LASSO penalty parameter.
    :param n_iter: Number of Gibbs iterations.
    :param seed: Random seed.
    :return: Dictionary with beta_samples, posterior_mean, posterior_median.

    References
    ----------
    Park, T. & Casella, G. (2008). *JASA*, 103(482), 681--686.
    """
    rng = np.random.default_rng(seed)
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, p = X_arr.shape

    beta = np.zeros(p)
    sigma2 = 1.0
    tau2 = np.ones(p)
    lam2 = lam ** 2

    beta_samples = np.empty((n_iter, p))
    XtX = X_arr.T @ X_arr
    Xty = X_arr.T @ y_arr

    for it in range(n_iter):
        D_inv = np.diag(1.0 / (tau2 + 1e-30))
        prec = XtX / sigma2 + D_inv
        cov = np.linalg.inv(prec)
        mean = cov @ (Xty / sigma2)
        beta = rng.multivariate_normal(mean, cov)

        resid = y_arr - X_arr @ beta
        post_a = (n - 1 + p) / 2.0
        post_b = 0.5 * float(resid @ resid) + 0.5 * float(np.sum(beta ** 2 / (tau2 + 1e-30)))
        sigma2 = 1.0 / rng.gamma(post_a, 1.0 / (post_b + 1e-30))

        for j in range(p):
            mu_inv_gauss = np.sqrt(lam2 * sigma2 / (beta[j] ** 2 + 1e-30))
            tau2[j] = 1.0 / _rinvgauss(rng, mu_inv_gauss, lam2)

        beta_samples[it] = beta

    return {
        "beta_samples": beta_samples,
        "posterior_mean": np.mean(beta_samples, axis=0).tolist(),
        "posterior_median": np.median(beta_samples, axis=0).tolist(),
        "lambda": lam,
        "n_iter": n_iter,
    }


def _rinvgauss(rng, mu, lam):
    v = rng.standard_normal() ** 2
    x = mu + (mu ** 2 * v - mu * np.sqrt(4 * mu * lam * v + mu ** 2 * v ** 2)) / (2 * lam)
    if x <= 0:
        x = mu ** 2 / (x + 1e-30) if x != 0 else mu
    u = rng.uniform()
    if u <= mu / (mu + x):
        return x
    return mu ** 2 / (x + 1e-30)


blasr = bayesian_lasso


def cheatsheet() -> str:
    return "bayesian_lasso({}) -> Bayesian LASSO regression."
