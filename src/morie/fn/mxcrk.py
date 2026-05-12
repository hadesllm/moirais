# morie.fn -- function file (hadesllm/morie)
"""Mixture cure model."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

__all__ = ["mxcrk"]


def mxcrk(
    time: np.ndarray,
    event: np.ndarray,
    X_cure: np.ndarray | None = None,
    X_surv: np.ndarray | None = None,
) -> dict:
    """Mixture cure model with logistic incidence and Weibull latency.

    S(t|X) = pi(X) + (1 - pi(X)) * S_u(t|X)

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    X_cure : array-like, optional
        Covariates for cure fraction (logistic) (n, p1).
    X_surv : array-like, optional
        Covariates for survival among uncured (n, p2).

    Returns
    -------
    dict
        cure_coefficients, survival_shape, survival_scale,
        cure_fraction, log_likelihood, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)

    p1 = 0
    if X_cure is not None:
        X_cure = np.asarray(X_cure, dtype=float)
        p1 = X_cure.shape[1]

    p2 = 0
    if X_surv is not None:
        X_surv = np.asarray(X_surv, dtype=float)
        p2 = X_surv.shape[1]

    def neg_loglik(params):
        gamma0 = params[0]
        gamma = params[1:1 + p1]
        log_k = params[1 + p1]
        log_lam = params[2 + p1]

        eta = gamma0 + (X_cure @ gamma if p1 > 0 else 0)
        pi = expit(eta)
        k = np.exp(log_k)
        lam = np.exp(log_lam)

        Su = np.exp(-(time / lam) ** k)
        fu = (k / lam) * (time / lam) ** (k - 1) * Su

        pop_surv = pi + (1 - pi) * Su
        pop_dens = (1 - pi) * fu

        ll = np.sum(event * np.log(np.maximum(pop_dens, 1e-300)))
        ll += np.sum((1 - event) * np.log(np.maximum(pop_surv, 1e-300)))
        return -ll if np.isfinite(ll) else 1e20

    x0 = np.zeros(2 + p1 + 1 + 1)
    x0[1 + p1] = 0.0
    x0[2 + p1] = np.log(np.median(time) + 1e-6)
    result = minimize(neg_loglik, x0, method="Nelder-Mead",
                      options={"maxiter": 5000})

    gamma0 = result.x[0]
    gamma = result.x[1:1 + p1]
    shape = np.exp(result.x[1 + p1])
    scale = np.exp(result.x[2 + p1])
    cure_frac = float(expit(gamma0))

    return {
        "cure_intercept": float(gamma0),
        "cure_coefficients": gamma,
        "survival_shape": float(shape),
        "survival_scale": float(scale),
        "cure_fraction": cure_frac,
        "log_likelihood": float(-result.fun),
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


mxcrk_fn = mxcrk


def cheatsheet() -> str:
    return "mxcrk(time, event) -> Mixture cure survival model."
