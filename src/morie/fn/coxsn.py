# morie.fn — function file (hadesllm/morie)
"""Cox-Snell residuals for survival model diagnostics."""

from __future__ import annotations

import numpy as np

__all__ = ["coxsn"]


def coxsn(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
) -> dict:
    """Cox-Snell residuals from a fitted Cox model.

    r_i = H_0(t_i) * exp(X_i * beta), where H_0 is the
    Breslow cumulative baseline hazard estimator.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times, shape (n,).
    event : array-like
        Event indicator (1=event, 0=censored), shape (n,).
    X : array-like
        Covariate matrix, shape (n, p).
    beta : array-like
        Fitted Cox coefficients, shape (p,).

    Returns
    -------
    dict
        residuals, baseline_cumhaz, times.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n = len(time)

    lp = X @ beta
    exp_lp = np.exp(lp)

    order = np.argsort(time, kind="stable")
    t_s = time[order]
    e_s = event[order]
    exp_lp_s = exp_lp[order]

    event_times = np.unique(t_s[e_s == 1])
    H0 = np.zeros(n)
    cum_h = 0.0

    for tj in event_times:
        at_risk = t_s >= tj
        dj = np.sum((t_s == tj) & (e_s == 1))
        denom = np.sum(exp_lp_s[at_risk])
        if denom > 0:
            cum_h += dj / denom
        H0[t_s <= tj] = cum_h

    inv_order = np.argsort(order)
    H0_orig = H0[inv_order]
    residuals = H0_orig * exp_lp

    return {
        "residuals": residuals,
        "baseline_cumhaz": H0_orig,
        "event": event,
    }


coxsn_fn = coxsn


def cheatsheet() -> str:
    return "coxsn(time, event, X, beta) -> Cox-Snell residuals."
