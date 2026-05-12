# morie.fn -- function file (hadesllm/morie)
"""Kaplan-Meier survival curve with confidence bands."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["kmsrv"]


def kmsrv(
    time: np.ndarray,
    event: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict:
    """Kaplan-Meier survival curve with pointwise CIs.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times.
    event : array-like
        Event indicator (1=event, 0=censored).
    alpha : float
        Significance level for CIs.

    Returns
    -------
    dict
        times, survival, se, ci_lower, ci_upper, median_survival,
        n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if time.shape[0] != event.shape[0]:
        raise ValueError("time and event must have the same length.")
    if time.shape[0] == 0:
        raise ValueError("Input arrays must not be empty.")

    order = np.argsort(time, kind="stable")
    t_s = time[order]
    e_s = event[order]

    unique_t = np.unique(t_s[e_s == 1])
    nt = len(unique_t)
    surv = np.ones(nt)
    se_arr = np.zeros(nt)
    z = norm.ppf(1 - alpha / 2)
    s = 1.0
    gw = 0.0

    for j, tj in enumerate(unique_t):
        nj = np.sum(t_s >= tj)
        dj = np.sum((t_s == tj) & (e_s == 1))
        if nj > 0:
            s *= 1 - dj / nj
            if nj > dj:
                gw += dj / (nj * (nj - dj))
        surv[j] = s
        se_arr[j] = s * np.sqrt(max(gw, 0))

    ci_lo = np.maximum(surv - z * se_arr, 0)
    ci_hi = np.minimum(surv + z * se_arr, 1)

    median_surv = np.nan
    below = np.where(surv <= 0.5)[0]
    if len(below) > 0:
        median_surv = float(unique_t[below[0]])

    return {
        "times": unique_t,
        "survival": surv,
        "se": se_arr,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "median_survival": median_surv,
        "n_obs": len(time),
        "n_events": int(np.sum(event)),
    }


kmsrv_fn = kmsrv


def cheatsheet() -> str:
    return "kmsrv(time, event) -> Kaplan-Meier survival curve with confidence bands."
