# morie.fn — function file (hadesllm/morie)
"""Piecewise exponential model."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def piecewise_exponential(
    times: np.ndarray | list,
    events: np.ndarray | list,
    breaks: np.ndarray | list | None = None,
) -> DescriptiveResult:
    """
    Fit a piecewise exponential model.

    Parameters
    ----------
    times : array-like
        Event/censoring times.
    events : array-like
        Event indicators (1 = event, 0 = censored).
    breaks : array-like, optional
        Cut points for intervals. Default: quartiles of event times.

    Returns
    -------
    DescriptiveResult
        extra has 'intervals', 'hazard_rates', 'n_events', 'exposure'.

    References
    ----------
    Friedman, M. (1982). Piecewise exponential models for survival
    data with covariates. *Ann Stat*, 10(1), 101-113.
    """
    t = np.asarray(times, dtype=float)
    d = np.asarray(events, dtype=int)
    if len(t) != len(d):
        raise ValueError("times and events must match.")

    if breaks is None:
        event_times = t[d == 1]
        if len(event_times) < 4:
            breaks = [np.median(t)]
        else:
            breaks = np.percentile(event_times, [25, 50, 75]).tolist()

    breaks = sorted(breaks)
    edges = [0.0] + list(breaks) + [float(np.max(t)) + 1]

    intervals = []
    hazard_rates = []
    n_events_list = []
    exposure_list = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_interval = t > lo
        exposure = np.sum(np.minimum(t[in_interval], hi) - lo)
        n_ev = np.sum((t >= lo) & (t < hi) & (d == 1))
        rate = n_ev / exposure if exposure > 0 else 0.0
        intervals.append((lo, hi))
        hazard_rates.append(float(rate))
        n_events_list.append(int(n_ev))
        exposure_list.append(float(exposure))

    return DescriptiveResult(
        name="piecewise_exponential",
        value=float(np.mean(hazard_rates)),
        extra={
            "intervals": intervals,
            "hazard_rates": hazard_rates,
            "n_events": n_events_list,
            "exposure": exposure_list,
        },
    )


pwexp = piecewise_exponential


def cheatsheet() -> str:
    return "piecewise_exponential({}) -> Piecewise exponential model."
