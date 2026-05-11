# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian logistic regression (Laplace approximation)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.optimize import minimize


def bayesian_logistic(
    X: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    *,
    prior_var: float = 100.0,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Bayesian logistic regression via Laplace approximation.

    Prior: beta ~ N(0, prior_var * I)
    Posterior approximated as N(beta_MAP, H^{-1}) where H is the Hessian at MAP.

    :param X: Design matrix (n, p).
    :param y: Binary response vector (n,).
    :param prior_var: Prior variance for each coefficient.
    :param prob: Credible interval probability.
    :return: Dictionary with MAP estimate, posterior covariance, and CIs.

    References
    ----------
    Bishop, C. (2006). *Pattern Recognition and Machine Learning*, Ch. 4.5.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, p = X_arr.shape

    prior_prec = np.eye(p) / prior_var

    def neg_log_posterior(beta):
        eta = X_arr @ beta
        eta_clip = np.clip(eta, -500, 500)
        ll = float(np.sum(y_arr * eta_clip - np.log1p(np.exp(eta_clip))))
        prior = -0.5 * float(beta @ prior_prec @ beta)
        return -(ll + prior)

    def grad(beta):
        eta = X_arr @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        g_ll = X_arr.T @ (y_arr - mu)
        g_prior = -prior_prec @ beta
        return -(g_ll + g_prior)

    beta0 = np.zeros(p)
    res = minimize(neg_log_posterior, beta0, jac=grad, method="L-BFGS-B")
    beta_map = res.x

    eta = X_arr @ beta_map
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
    W = np.diag(mu * (1 - mu))
    H = X_arr.T @ W @ X_arr + prior_prec
    cov = np.linalg.inv(H)

    from scipy import stats as st

    alpha = 1 - prob
    se = np.sqrt(np.diag(cov))
    z_lo = st.norm.ppf(alpha / 2)
    z_hi = st.norm.ppf(1 - alpha / 2)

    return {
        "beta_map": beta_map.tolist(),
        "posterior_cov": cov.tolist(),
        "se": se.tolist(),
        "ci_lower": (beta_map + z_lo * se).tolist(),
        "ci_upper": (beta_map + z_hi * se).tolist(),
        "log_marginal_approx": float(-res.fun + 0.5 * p * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(cov))),
        "converged": bool(res.success),
    }


blogt = bayesian_logistic


def cheatsheet() -> str:
    return "bayesian_logistic({}) -> Bayesian logistic regression (Laplace approximation)."
