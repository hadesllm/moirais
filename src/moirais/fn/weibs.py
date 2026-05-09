"""Weibull survival model (AFT parameterization)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["weibs"]


def weibs(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray | None = None,
) -> dict:
    """Weibull AFT survival model via maximum likelihood.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    X : array-like, optional
        Covariate matrix (n, p). If None, fits intercept-only.

    Returns
    -------
    dict
        shape, scale, coefficients, se, log_likelihood, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)
    if X is not None:
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
    else:
        X = np.ones((n, 0))
        p = 0

    def neg_loglik(params):
        log_k = params[0]
        mu = params[1]
        beta = params[2:] if p > 0 else np.array([])
        k = np.exp(log_k)
        eta = mu + (X @ beta if p > 0 else 0)
        lam = np.exp(-eta)
        ll = np.sum(event * (np.log(k) + (k - 1) * np.log(lam * time) + np.log(lam)))
        ll -= np.sum((lam * time) ** k)
        return -ll

    x0 = np.zeros(2 + p)
    result = minimize(neg_loglik, x0, method="L-BFGS-B")

    log_k = result.x[0]
    mu = result.x[1]
    beta = result.x[2:] if p > 0 else np.array([])
    shape = np.exp(log_k)
    scale = np.exp(mu)

    try:
        h = np.zeros(len(result.x))
        eps = 1e-5
        for i in range(len(result.x)):
            e_i = np.zeros(len(result.x))
            e_i[i] = eps
            h[i] = (neg_loglik(result.x + e_i) - 2 * neg_loglik(result.x) + neg_loglik(result.x - e_i)) / eps ** 2
        se = 1.0 / np.sqrt(np.maximum(h[2:], 1e-10)) if p > 0 else np.array([])
    except Exception:
        se = np.full(p, np.nan)

    return {
        "shape": float(shape),
        "scale": float(scale),
        "coefficients": beta,
        "se": se,
        "log_likelihood": float(-result.fun),
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


weibs_fn = weibs


def cheatsheet() -> str:
    return "weibs(time, event, X) -> Weibull AFT survival model."
