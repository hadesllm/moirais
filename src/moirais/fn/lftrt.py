# moirais.fn — function file (hadesllm/moirais)
"""Left truncation adjustment for survival analysis."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["lftrt"]


def lftrt(
    entry: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict:
    """Kaplan-Meier estimator with left truncation (delayed entry).

    Parameters
    ----------
    entry : array-like
        Left truncation (entry) times (n,).
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    alpha : float
        Significance level for CIs.

    Returns
    -------
    dict
        times, survival, se, ci_lower, ci_upper, n_obs, n_events.
    """
    entry = np.asarray(entry, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)

    unique_t = np.sort(np.unique(time[event == 1]))
    nt = len(unique_t)
    surv = np.ones(nt)
    se_arr = np.zeros(nt)
    z = norm.ppf(1 - alpha / 2)
    s = 1.0
    gw = 0.0

    for j, tj in enumerate(unique_t):
        nj = np.sum((entry < tj) & (time >= tj))
        dj = np.sum((time == tj) & (event == 1))
        if nj > 0:
            s *= 1 - dj / nj
            if nj > dj:
                gw += dj / (nj * (nj - dj))
        surv[j] = s
        se_arr[j] = s * np.sqrt(max(gw, 0))

    ci_lo = np.maximum(surv - z * se_arr, 0)
    ci_hi = np.minimum(surv + z * se_arr, 1)

    return {
        "times": unique_t,
        "survival": surv,
        "se": se_arr,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


lftrt_fn = lftrt


def cheatsheet() -> str:
    return "lftrt(entry, time, event) -> Left-truncated survival estimator."
