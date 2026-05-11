# morie.fn — function file (hadesllm/morie)
"""Exponential survival model."""

from __future__ import annotations

import numpy as np

__all__ = ["expsv"]


def expsv(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray | None = None,
) -> dict:
    """Exponential (constant hazard) survival model.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    X : array-like, optional
        Covariate matrix (n, p).

    Returns
    -------
    dict
        rate, coefficients, se, log_likelihood, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)

    if X is None:
        d = np.sum(event)
        T_total = np.sum(time)
        rate = d / T_total if T_total > 0 else 0.0
        ll = d * np.log(rate + 1e-300) - rate * T_total
        se_rate = rate / np.sqrt(max(d, 1))
        return {
            "rate": float(rate),
            "coefficients": np.array([]),
            "se": np.array([]),
            "log_likelihood": float(ll),
            "n_obs": n,
            "n_events": int(d),
            "se_rate": float(se_rate),
        }

    X = np.asarray(X, dtype=float)
    p = X.shape[1]

    from scipy.optimize import minimize

    def neg_loglik(params):
        mu = params[0]
        beta = params[1:]
        eta = mu + X @ beta
        lam = np.exp(eta)
        ll = np.sum(event * eta) - np.sum(lam * time)
        return -ll

    x0 = np.zeros(1 + p)
    x0[0] = np.log(np.sum(event) / np.sum(time) + 1e-300)
    result = minimize(neg_loglik, x0, method="L-BFGS-B")

    mu = result.x[0]
    beta = result.x[1:]
    rate = np.exp(mu)

    try:
        eps = 1e-5
        hess = np.zeros(p)
        for i in range(p):
            e_i = np.zeros(1 + p)
            e_i[1 + i] = eps
            hess[i] = (neg_loglik(result.x + e_i) - 2 * neg_loglik(result.x) + neg_loglik(result.x - e_i)) / eps ** 2
        se = 1.0 / np.sqrt(np.maximum(hess, 1e-10))
    except Exception:
        se = np.full(p, np.nan)

    return {
        "rate": float(rate),
        "coefficients": beta,
        "se": se,
        "log_likelihood": float(-result.fun),
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


expsv_fn = expsv


def cheatsheet() -> str:
    return "expsv(time, event, X) -> Exponential survival model."
