# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian linear regression (conjugate normal-inverse-gamma)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy import stats


def bayesian_linear_regression(
    X: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    *,
    prior_mu: Union[list, np.ndarray, None] = None,
    prior_Lambda: Union[list, np.ndarray, None] = None,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Bayesian linear regression with normal-inverse-gamma conjugate prior.

    Model: y | X, beta, sigma^2 ~ N(X beta, sigma^2 I)
    Prior: beta | sigma^2 ~ N(mu_0, sigma^2 Lambda_0^{-1}), sigma^2 ~ IG(a, b)

    :param X: Design matrix (n, p).
    :param y: Response vector (n,).
    :param prior_mu: Prior mean for beta (p,). Default zeros.
    :param prior_Lambda: Prior precision matrix (p, p). Default 0.01*I.
    :param prior_a: Shape of inverse-gamma prior on sigma^2.
    :param prior_b: Scale of inverse-gamma prior on sigma^2.
    :param prob: Credible interval probability.
    :return: Dictionary with posterior parameters and summaries.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed., Ch. 14.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, p = X_arr.shape

    if prior_mu is None:
        prior_mu = np.zeros(p)
    else:
        prior_mu = np.asarray(prior_mu, dtype=float)

    if prior_Lambda is None:
        prior_Lambda = 0.01 * np.eye(p)
    else:
        prior_Lambda = np.asarray(prior_Lambda, dtype=float)

    XtX = X_arr.T @ X_arr
    Xty = X_arr.T @ y_arr

    post_Lambda = prior_Lambda + XtX
    post_Lambda_inv = np.linalg.inv(post_Lambda)
    post_mu = post_Lambda_inv @ (prior_Lambda @ prior_mu + Xty)

    post_a = prior_a + n / 2.0
    resid = y_arr - X_arr @ post_mu
    quad = float((prior_mu - post_mu).T @ prior_Lambda @ (prior_mu - post_mu))
    post_b = prior_b + 0.5 * (float(resid @ resid) + quad)

    post_sigma2_mean = post_b / (post_a - 1) if post_a > 1 else float("inf")

    alpha = 1 - prob
    ci_lower = []
    ci_upper = []
    scale_factor = post_b / post_a
    for j in range(p):
        se = float(np.sqrt(scale_factor * post_Lambda_inv[j, j]))
        lo = post_mu[j] + se * stats.t.ppf(alpha / 2, df=2 * post_a)
        hi = post_mu[j] + se * stats.t.ppf(1 - alpha / 2, df=2 * post_a)
        ci_lower.append(float(lo))
        ci_upper.append(float(hi))

    return {
        "posterior_mean": post_mu.tolist(),
        "posterior_precision": post_Lambda.tolist(),
        "posterior_a": float(post_a),
        "posterior_b": float(post_b),
        "posterior_sigma2_mean": float(post_sigma2_mean),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
        "p": p,
    }


bmlnr = bayesian_linear_regression


def cheatsheet() -> str:
    return "bayesian_linear_regression({}) -> Bayesian linear regression (conjugate normal-inverse-gamma)."
