# moirais.fn — function file (hadesllm/moirais)
"""Nelson-Aalen cumulative hazard estimator."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["naest"]


def naest(
    time: np.ndarray,
    event: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict:
    """Nelson-Aalen cumulative hazard estimator.

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
        times, cumulative_hazard, se, ci_lower, ci_upper, n_obs, n_events.
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
    cum_haz = np.zeros(nt)
    var_arr = np.zeros(nt)
    z = norm.ppf(1 - alpha / 2)
    H = 0.0
    V = 0.0

    for j, tj in enumerate(unique_t):
        nj = np.sum(t_s >= tj)
        dj = np.sum((t_s == tj) & (e_s == 1))
        if nj > 0:
            H += dj / nj
            V += dj / (nj ** 2)
        cum_haz[j] = H
        var_arr[j] = V

    se = np.sqrt(var_arr)
    ci_lo = np.maximum(cum_haz - z * se, 0)
    ci_hi = cum_haz + z * se

    return {
        "times": unique_t,
        "cumulative_hazard": cum_haz,
        "se": se,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_obs": len(time),
        "n_events": int(np.sum(event)),
    }


naest_fn = naest


def cheatsheet() -> str:
    return "naest(time, event) -> Nelson-Aalen cumulative hazard estimator."
