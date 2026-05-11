# morie.fn — function file (hadesllm/morie)
"""Piecewise constant hazard rate estimator."""

from __future__ import annotations

import numpy as np

__all__ = ["pchzr"]


def pchzr(
    time: np.ndarray,
    event: np.ndarray,
    *,
    breaks: np.ndarray | None = None,
    n_intervals: int = 5,
) -> dict:
    """Piecewise constant hazard rate estimation.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    breaks : array-like, optional
        Interval boundaries. Default: quantiles.
    n_intervals : int
        Number of intervals if breaks not given.

    Returns
    -------
    dict
        breaks, hazard_rates, n_events_per_interval,
        exposure_per_interval, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)

    if breaks is None:
        q = np.linspace(0, 100, n_intervals + 1)
        breaks = np.unique(np.percentile(time, q))
        if breaks[0] > 0:
            breaks = np.concatenate([[0], breaks])
    else:
        breaks = np.asarray(breaks, dtype=float)

    K = len(breaks) - 1
    haz = np.zeros(K)
    d_k = np.zeros(K, dtype=int)
    exp_k = np.zeros(K)

    for k in range(K):
        lo, hi = breaks[k], breaks[k + 1]
        in_int = (time > lo) & (time <= hi)
        at_risk = time > lo
        d_k[k] = int(np.sum(in_int & (event == 1)))
        exposure = np.sum(np.minimum(time[at_risk], hi) - lo)
        exp_k[k] = exposure
        haz[k] = d_k[k] / exposure if exposure > 0 else 0.0

    return {
        "breaks": breaks,
        "hazard_rates": haz,
        "n_events_per_interval": d_k,
        "exposure_per_interval": exp_k,
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


pchzr_fn = pchzr


def cheatsheet() -> str:
    return "pchzr(time, event) -> Piecewise constant hazard rate."
