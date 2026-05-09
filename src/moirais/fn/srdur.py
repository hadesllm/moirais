"""Survival duration statistics (median, percentiles)."""

from __future__ import annotations

import numpy as np

__all__ = ["srdur"]


def srdur(
    time: np.ndarray,
    event: np.ndarray,
    *,
    percentiles: list | None = None,
) -> dict:
    """Survival duration statistics from Kaplan-Meier.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    percentiles : list, optional
        Percentiles to compute. Default: [25, 50, 75].

    Returns
    -------
    dict
        median_survival, percentile_times, mean_restricted,
        n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)

    if percentiles is None:
        percentiles = [25, 50, 75]

    order = np.argsort(time, kind="stable")
    t_s = time[order]
    e_s = event[order]
    unique_t = np.unique(t_s[e_s == 1])

    surv = []
    s = 1.0
    for tj in unique_t:
        nj = np.sum(t_s >= tj)
        dj = np.sum((t_s == tj) & (e_s == 1))
        if nj > 0:
            s *= 1 - dj / nj
        surv.append(s)
    surv = np.array(surv)

    pct_times = {}
    for p in percentiles:
        target = 1 - p / 100
        idx = np.where(surv <= target)[0]
        if len(idx) > 0:
            pct_times[p] = float(unique_t[idx[0]])
        else:
            pct_times[p] = np.nan

    mean_rst = 0.0
    if len(unique_t) > 0:
        all_t = np.concatenate([[0], unique_t])
        all_s = np.concatenate([[1.0], surv])
        for i in range(len(all_t) - 1):
            mean_rst += all_s[i] * (all_t[i + 1] - all_t[i])

    return {
        "median_survival": pct_times.get(50, np.nan),
        "percentile_times": pct_times,
        "mean_restricted": float(mean_rst),
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


srdur_fn = srdur


def cheatsheet() -> str:
    return "srdur(time, event) -> Survival duration statistics (median, percentiles)."
