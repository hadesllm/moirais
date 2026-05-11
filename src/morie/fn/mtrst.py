# morie.fn — function file (hadesllm/morie)
"""Martingale residuals for Cox proportional hazards model."""

from __future__ import annotations

import numpy as np

__all__ = ["mtrst"]


def mtrst(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
) -> dict:
    """Martingale residuals from a fitted Cox model.

    M_i = delta_i - H_0(t_i) * exp(X_i * beta)

    Parameters
    ----------
    time : array-like
        Observed event/censoring times.
    event : array-like
        Event indicator (1=event, 0=censored).
    X : array-like
        Covariate matrix (n, p).
    beta : array-like
        Fitted Cox coefficients (p,).

    Returns
    -------
    dict
        residuals, mean_residual.
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
    residuals = event - H0_orig * exp_lp

    return {
        "residuals": residuals,
        "mean_residual": float(np.mean(residuals)),
    }


mtrst_fn = mtrst


def cheatsheet() -> str:
    return "mtrst(time, event, X, beta) -> Martingale residuals for Cox model."
