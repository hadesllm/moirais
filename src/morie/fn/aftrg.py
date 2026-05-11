# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Accelerated failure time regression."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = ["aftrg"]


def aftrg(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    *,
    dist: str = "lognormal",
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""
    Fit an accelerated failure time (AFT) model.

    The AFT model specifies :math:`\log T = X^T \beta + \sigma W`
    where :math:`W` follows a known distribution.

    :param time: Event/censoring times, shape (n,). Must be positive.
    :param event: Event indicators (1=event, 0=censored), shape (n,).
    :param X: Covariate matrix, shape (n, p).
    :param dist: Error distribution: ``"lognormal"`` (default), ``"loglogistic"``,
        ``"weibull"``.
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``beta``, ``sigma``, ``se``, ``ci_lower``, ``ci_upper``,
        ``log_likelihood``, ``n``, ``dist``.
    :raises ValueError: If arrays are mismatched or times non-positive.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 12. Springer.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n == 0:
        raise ValueError("Arrays must be non-empty.")
    if np.any(time <= 0):
        raise ValueError("time must be positive.")

    log_t = np.log(time)
    X_aug = np.column_stack([np.ones(n), X])

    if dist == "lognormal":
        log_pdf = lambda z: stats.norm.logpdf(z)
        log_sf = lambda z: stats.norm.logsf(z)
    elif dist == "loglogistic":
        log_pdf = lambda z: stats.logistic.logpdf(z)
        log_sf = lambda z: stats.logistic.logsf(z)
    elif dist == "weibull":
        log_pdf = lambda z: -(z + np.exp(z))
        log_sf = lambda z: -np.exp(z)
    else:
        raise ValueError(f"dist must be 'lognormal', 'loglogistic', or 'weibull', got '{dist}'.")

    def neg_loglik(params):
        beta = params[:p + 1]
        log_sigma = params[p + 1]
        sigma = np.exp(log_sigma)
        resid = (log_t - X_aug @ beta) / sigma
        ll = np.sum(event * (log_pdf(resid) - log_sigma) + (1 - event) * log_sf(resid))
        return -ll

    theta0 = np.zeros(p + 2)
    beta_init = np.linalg.lstsq(X_aug[event == 1], log_t[event == 1], rcond=None)[0]
    theta0[:p + 1] = beta_init

    result = optimize.minimize(neg_loglik, theta0, method="BFGS")
    beta_hat = result.x[:p + 1]
    sigma_hat = np.exp(result.x[p + 1])
    log_ll = -result.fun

    eps = 1e-5
    d = len(result.x)
    hessian = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            ei, ej = np.zeros(d), np.zeros(d)
            ei[i] = eps
            ej[j] = eps
            hessian[i, j] = (neg_loglik(result.x + ei + ej) - neg_loglik(result.x + ei - ej) - neg_loglik(result.x - ei + ej) + neg_loglik(result.x - ei - ej)) / (4 * eps ** 2)

    try:
        var = np.linalg.inv(hessian)
        se = np.sqrt(np.maximum(np.diag(var), 0))
    except np.linalg.LinAlgError:
        se = np.full(d, np.nan)

    se_beta = se[:p + 1]
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        "beta": beta_hat,
        "sigma": float(sigma_hat),
        "se": se_beta,
        "ci_lower": beta_hat - z * se_beta,
        "ci_upper": beta_hat + z * se_beta,
        "log_likelihood": float(log_ll),
        "n": n,
        "dist": dist,
    }


def cheatsheet() -> str:
    return "aftrg({time, event, X}) -> Accelerated failure time regression."
