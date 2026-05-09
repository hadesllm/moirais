"""Smoothed hazard function (kernel estimation)."""

from __future__ import annotations

import numpy as np

__all__ = ["smhaz"]


def smhaz(
    time: np.ndarray,
    event: np.ndarray,
    *,
    bandwidth: float | None = None,
    n_grid: int = 100,
) -> dict:
    """Kernel-smoothed hazard function estimator.

    Uses Epanechnikov kernel with bandwidth selection via
    Silverman's rule of thumb adapted for hazard estimation.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    bandwidth : float, optional
        Kernel bandwidth. Default: Silverman's rule.
    n_grid : int
        Number of grid points for evaluation.

    Returns
    -------
    dict
        grid, hazard, cumulative_hazard, bandwidth, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)
    event_times = time[event == 1]

    if bandwidth is None:
        std = np.std(event_times) if len(event_times) > 1 else np.std(time)
        bandwidth = 0.9 * std * len(event_times) ** (-0.2) if len(event_times) > 0 else 1.0
    bandwidth = max(bandwidth, 1e-6)

    grid = np.linspace(np.min(time) * 0.9, np.max(time) * 1.1, n_grid)
    haz = np.zeros(n_grid)

    for i, t in enumerate(grid):
        at_risk = np.sum(time >= t)
        if at_risk == 0:
            continue
        u = (event_times - t) / bandwidth
        mask = np.abs(u) <= 1
        K = np.where(mask, 0.75 * (1 - u ** 2), 0)
        haz[i] = np.sum(K) / (at_risk * bandwidth)

    dt = grid[1] - grid[0] if n_grid > 1 else 1.0
    cum_haz = np.cumsum(haz) * dt

    return {
        "grid": grid,
        "hazard": haz,
        "cumulative_hazard": cum_haz,
        "bandwidth": float(bandwidth),
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


smhaz_fn = smhaz


def cheatsheet() -> str:
    return "smhaz(time, event) -> Kernel-smoothed hazard function."
