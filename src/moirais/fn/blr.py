# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian linear regression (conjugate Normal-Inverse-Gamma)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_linear_regression(
    X: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    *,
    prior_mean: Union[float, np.ndarray] = 0.0,
    prior_precision: float = 0.01,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Bayesian linear regression with a conjugate Normal-Inverse-Gamma prior.

    Prior on coefficients: beta | sigma^2 ~ N(prior_mean, sigma^2 / prior_precision * I).
    Prior on variance: sigma^2 ~ InvGamma(prior_a, prior_b).

    Posterior for beta (marginalised over sigma^2) follows a multivariate t
    distribution. Posterior mean and covariance of beta are returned.

    :param X: Design matrix of shape (n, p). An intercept column is NOT
        added automatically.
    :param y: Response vector of length n.
    :param prior_mean: Prior mean for beta (scalar broadcast to p, or array of length p).
    :param prior_precision: Scalar precision multiplier (Lambda_0 = prior_precision * I).
    :param prior_a: Shape of Inverse-Gamma prior on sigma^2.
    :param prior_b: Scale of Inverse-Gamma prior on sigma^2.
    :param alpha: Significance level for credible intervals.
    :return: Dictionary with posterior_mean, posterior_cov, sigma2_mean,
        sigma2_mode, credible_intervals (list of tuples per coefficient).
    :raises ValueError: If dimensions are incompatible.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.),
    Chapter 14. CRC Press.

    Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*.
    MIT Press, Section 7.6.
    """
    Xm = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float)
    if Xm.ndim == 1:
        Xm = Xm.reshape(-1, 1)
    n, p = Xm.shape
    if len(yv) != n:
        raise ValueError(f"X has {n} rows but y has {len(yv)} elements.")

    # Prior
    if np.isscalar(prior_mean):
        beta_0 = np.full(p, float(prior_mean))
    else:
        beta_0 = np.asarray(prior_mean, dtype=float)
    Lambda_0 = prior_precision * np.eye(p)

    # Posterior parameters
    Lambda_n = Lambda_0 + Xm.T @ Xm
    Lambda_n_inv = np.linalg.inv(Lambda_n)
    beta_n = Lambda_n_inv @ (Lambda_0 @ beta_0 + Xm.T @ yv)

    a_n = prior_a + n / 2.0
    residual = yv - Xm @ beta_n
    b_n = prior_b + 0.5 * (float(residual @ residual) + float((beta_n - beta_0) @ Lambda_0 @ (beta_n - beta_0)))

    sigma2_mean = b_n / (a_n - 1.0) if a_n > 1 else float("inf")
    sigma2_mode = b_n / (a_n + 1.0)

    # Posterior covariance of beta (integrating out sigma^2)
    # Var(beta | data) = b_n / (a_n - 1) * Lambda_n_inv  (when a_n > 1)
    if a_n > 1:
        post_cov = (b_n / (a_n - 1.0)) * Lambda_n_inv
    else:
        post_cov = Lambda_n_inv  # fallback

    # Credible intervals (using t-distribution with 2*a_n degrees of freedom)
    from scipy import stats as _st

    df = 2.0 * a_n
    post_sd = np.sqrt(np.diag(post_cov))
    t_crit = _st.t.ppf(1.0 - alpha / 2.0, df)
    ci_list = [(float(beta_n[j] - t_crit * post_sd[j]), float(beta_n[j] + t_crit * post_sd[j])) for j in range(p)]

    return {
        "posterior_mean": beta_n,
        "posterior_cov": post_cov,
        "sigma2_mean": float(sigma2_mean),
        "sigma2_mode": float(sigma2_mode),
        "credible_intervals": ci_list,
        "posterior_a": float(a_n),
        "posterior_b": float(b_n),
    }


blr = bayesian_linear_regression


def cheatsheet() -> str:
    return "bayesian_linear_regression({}) -> Bayesian linear regression (conjugate Normal-Inverse-Gamma)."
