# moirais.fn — function file (hadesllm/moirais)
"""Gompertz survival model."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["gompr"]


def gompr(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray | None = None,
) -> dict:
    """Gompertz proportional hazards survival model.

    h(t|X) = lambda * exp(gamma * t + X * beta)

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
        lambda_, gamma, coefficients, log_likelihood, n_obs, n_events.
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
        log_lam = params[0]
        gamma = params[1]
        beta = params[2:] if p > 0 else np.array([])
        lam = np.exp(log_lam)
        eta = X @ beta if p > 0 else 0
        exp_eta = np.exp(eta)
        h = lam * np.exp(gamma * time) * exp_eta
        if gamma == 0:
            H = lam * time * exp_eta
        else:
            H = (lam / gamma) * (np.exp(gamma * time) - 1) * exp_eta
        ll = np.sum(event * np.log(h + 1e-300)) - np.sum(H)
        return -ll if np.isfinite(ll) else 1e20

    x0 = np.zeros(2 + p)
    x0[0] = np.log(np.sum(event) / np.sum(time) + 1e-300)
    result = minimize(neg_loglik, x0, method="Nelder-Mead",
                      options={"maxiter": 5000})

    lam = np.exp(result.x[0])
    gamma = result.x[1]
    beta = result.x[2:] if p > 0 else np.array([])

    return {
        "lambda_": float(lam),
        "gamma": float(gamma),
        "coefficients": beta,
        "log_likelihood": float(-result.fun),
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


gompr_fn = gompr


def cheatsheet() -> str:
    return "gompr(time, event, X) -> Gompertz survival model."
