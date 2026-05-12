# morie.fn -- function file (hadesllm/morie)
"""Generalized gamma survival model."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammainc, gammaln
from scipy.stats import norm as sp_norm

__all__ = ["gngsv"]


def gngsv(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray | None = None,
) -> dict:
    """Generalized gamma AFT survival model.

    Nests exponential (Q=1, k=1), Weibull (k=1), log-normal (Q=0).

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
        mu, sigma, Q, coefficients, log_likelihood, n_obs, n_events.
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
        Q = params[2]
        beta = params[3:] if p > 0 else np.array([])
        sigma = np.exp(log_sig)
        eta = mu + (X @ beta if p > 0 else 0)
        z = (log_t - eta) / sigma

        if abs(Q) < 1e-6:
            ll_event = sp_norm.logpdf(z) - np.log(sigma) - log_t
            ll_cens = sp_norm.logsf(z)
        else:
            k = Q ** (-2)
            w = np.sign(Q) * z
            u = k * np.exp(abs(Q) * w)
            ll_event = (np.log(abs(Q)) + k * np.log(k) - gammaln(k) +
                        k * abs(Q) * w - u - np.log(sigma) - log_t)
            surv = np.where(Q > 0, 1 - gammainc(k, u), gammainc(k, u))
            surv = np.clip(surv, 1e-300, 1)
            ll_cens = np.log(surv)

        ll = np.sum(event * ll_event) + np.sum((1 - event) * ll_cens)
        return -ll if np.isfinite(ll) else 1e20

    x0 = np.zeros(3 + p)
    x0[0] = np.mean(log_t)
    x0[1] = np.log(np.std(log_t) + 1e-6)
    x0[2] = 1.0
    result = minimize(neg_loglik, x0, method="Nelder-Mead",
                      options={"maxiter": 5000})

    mu = result.x[0]
    sigma = np.exp(result.x[1])
    Q = result.x[2]
    beta = result.x[3:] if p > 0 else np.array([])

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "Q": float(Q),
        "coefficients": beta,
        "log_likelihood": float(-result.fun),
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


gngsv_fn = gngsv


def cheatsheet() -> str:
    return "gngsv(time, event, X) -> Generalized gamma survival model."
