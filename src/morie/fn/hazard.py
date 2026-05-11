# morie.fn — function file (hadesllm/morie)
"""Hazard rate function."""

from __future__ import annotations

import numpy as np

from ._containers import SurvivalResult


def hazard_rate(time, event, *, bandwidth: float | None = None) -> SurvivalResult:
    """Kernel-smoothed hazard rate estimate.

    Parameters
    ----------
    time : array-like
        Observed survival times.
    event : array-like
        Event indicator (1 = event, 0 = censored).
    bandwidth : float, optional
        Gaussian kernel bandwidth. Default: Silverman's rule.

    Returns
    -------
    SurvivalResult
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    mask = np.isfinite(time)
    time, event = time[mask], event[mask]
    if bandwidth is None:
        bandwidth = max(1.06 * time.std() * len(time) ** (-0.2), 1e-6)
    grid = np.linspace(time.min(), time.max(), 100)
    h = np.zeros(len(grid))
    for i, t in enumerate(grid):
        w = np.exp(-0.5 * ((time - t) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
        d = np.sum(w * event)
        n = np.sum(w * (time >= t))
        h[i] = d / n if n > 0 else 0
    return SurvivalResult(
        name="Hazard rate",
        times=grid,
        survival=h,
        n_events=int(event.sum()),
        n_censored=int((event == 0).sum()),
        extra={"bandwidth": bandwidth},
    )


hazard = hazard_rate


def cheatsheet() -> str:
    return "hazard_rate({}) -> Hazard rate function."
