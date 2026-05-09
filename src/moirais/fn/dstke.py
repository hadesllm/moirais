# moirais.fn — function file (hadesllm/moirais)
"""Hazard function estimation. 'I keep my promises.' -- Deathstroke"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def hazard_kernel(
    data: pd.DataFrame,
    *,
    time: str = "time",
    event: str = "event",
    bandwidth: float | None = None,
    n_grid: int = 100,
) -> DescriptiveResult:
    """Kernel-smoothed hazard rate estimation (Epanechnikov kernel).

    Estimates the hazard function h(t) = f(t) / S(t) non-parametrically
    using kernel density estimation on the event times.

    Parameters
    ----------
    data : DataFrame
        Survival data with time and event indicator columns.
    time : str
        Time column.
    event : str
        Event indicator (1 = event, 0 = censored).
    bandwidth : float or None
        Kernel bandwidth.  Default uses Silverman's rule.
    n_grid : int
        Number of evaluation points.

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with ``'time_grid'`` and ``'hazard'``.
    """
    _validate_df(data, time, event)
    df = data[[time, event]].dropna()
    t = df[time].to_numpy(dtype=float)
    e = df[event].to_numpy(dtype=int)
    n = len(t)
    if n < 5:
        raise ValueError("Need at least 5 observations")
    event_times = t[e == 1]
    if len(event_times) == 0:
        raise ValueError("No events observed")
    if bandwidth is None:
        bandwidth = 1.06 * np.std(event_times) * len(event_times) ** (-0.2)
        bandwidth = max(bandwidth, 1e-6)
    grid = np.linspace(t.min(), t.max(), n_grid)
    hazard = np.zeros(n_grid)
    for i, g in enumerate(grid):
        u = (event_times - g) / bandwidth
        kernel = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
        n_at_risk = np.sum(t >= g)
        if n_at_risk > 0:
            hazard[i] = np.sum(kernel) / (bandwidth * n_at_risk)
    cum_hazard = np.cumsum(hazard) * (grid[1] - grid[0]) if len(grid) > 1 else np.zeros(1)
    return DescriptiveResult(
        name="Kernel-smoothed hazard rate",
        value={"time_grid": grid.tolist(), "hazard": hazard.tolist()},
        extra={
            "n": n,
            "n_events": int(e.sum()),
            "bandwidth": float(bandwidth),
            "cumulative_hazard": cum_hazard.tolist(),
            "max_hazard": float(np.max(hazard)),
            "mean_event_time": float(np.mean(event_times)),
        },
    )


dstke = hazard_kernel


def cheatsheet() -> str:
    return "hazard_kernel({}) -> Hazard function estimation. 'I keep my promises.' -- Deathst"
