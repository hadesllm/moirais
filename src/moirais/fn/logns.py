# moirais.fn — function file (hadesllm/moirais)
"""Log-normal survival model."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as sp_norm

__all__ = ["logns"]


def logns(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray | None = None,
) -> dict:
    """Log-normal AFT survival model via maximum likelihood.

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
        mu, sigma, coefficients, log_likelihood, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)
    log_t = np.log(np.maximum(time, 1e-300))
    if X is not None:
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
    else:
        X = np.ones((n, 0))
        p = 0

    def neg_loglik(params):
        mu = params[0]
        log_sig = params[1]
        beta = params[2:] if p > 0 else np.array([])
        sigma = np.exp(log_sig)
        eta = mu + (X @ beta if p > 0 else 0)
        z = (log_t - eta) / sigma
        ll = np.sum(event * (sp_norm.logpdf(z) - np.log(sigma) - log_t))
        ll += np.sum((1 - event) * sp_norm.logsf(z))
        return -ll

    x0 = np.zeros(2 + p)
    x0[0] = np.mean(log_t)
    x0[1] = np.log(np.std(log_t) + 1e-6)
    result = minimize(neg_loglik, x0, method="L-BFGS-B")

    mu = result.x[0]
    sigma = np.exp(result.x[1])
    beta = result.x[2:] if p > 0 else np.array([])

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "coefficients": beta,
        "log_likelihood": float(-result.fun),
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


logns_fn = logns


def cheatsheet() -> str:
    return "logns(time, event, X) -> Log-normal AFT survival model."
