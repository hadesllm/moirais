"""Solve Newton's law of cooling: dT/dt = -k(T - T_ambient)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def newton_cooling(
    t_initial: float,
    t_ambient: float,
    *,
    k: float | None = None,
    times: np.ndarray | None = None,
    observed_temps: np.ndarray | None = None,
    observed_times: np.ndarray | None = None,
    t_max: float = 100.0,
    n_points: int = 200,
) -> DescriptiveResult:
    """Solve Newton's law of cooling: dT/dt = -k(T - T_ambient).

    If k is unknown, estimates it from observed data via least-squares.

    Parameters
    ----------
    t_initial : float
        Initial temperature.
    t_ambient : float
        Ambient (environment) temperature.
    k : float, optional
        Cooling constant. Estimated from data if not given.
    times : ndarray, optional
        Times at which to evaluate. Defaults to linspace(0, t_max).
    observed_temps : ndarray, optional
        Observed temperatures for fitting k.
    observed_times : ndarray, optional
        Times of observations.
    t_max : float
        Maximum time for evaluation.
    n_points : int
        Number of evaluation points.

    Returns
    -------
    DescriptiveResult
        With ``value`` = predicted temperatures and ``extra``.
    """
    if k is None:
        if observed_temps is None or observed_times is None:
            raise ValueError("Provide k or (observed_temps, observed_times)")
        t_obs = np.asarray(observed_temps, dtype=float).ravel()
        t_times = np.asarray(observed_times, dtype=float).ravel()
        if len(t_obs) < 2:
            raise ValueError("Need at least 2 observations to estimate k")
        diff = t_obs - t_ambient
        diff = np.where(np.abs(diff) < 1e-10, 1e-10, diff)
        log_diff = np.log(np.abs(diff / (t_initial - t_ambient + 1e-10)))
        slope, _ = np.polyfit(t_times, log_diff, 1)
        k = -slope
        if k < 0:
            k = abs(k)

    if times is None:
        times = np.linspace(0, t_max, n_points)
    else:
        times = np.asarray(times, dtype=float).ravel()

    T = t_ambient + (t_initial - t_ambient) * np.exp(-k * times)

    half_life = np.log(2) / k if k > 0 else float("inf")
    time_to_1pct = np.log(100) / k if k > 0 else float("inf")

    return DescriptiveResult(
        name="newton_cooling",
        value=T,
        extra={
            "k": float(k),
            "t_initial": t_initial,
            "t_ambient": t_ambient,
            "half_life": float(half_life),
            "time_to_1pct": float(time_to_1pct),
            "times": times,
        },
    )


newcoo = newton_cooling


def cheatsheet() -> str:
    return "newton_cooling({}) -> Newton's law of cooling."
