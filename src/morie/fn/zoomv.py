"""Velocity profile estimation. 'You can't lock up the darkness.' -- Zoom"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _extract_col


def velocity_profile(
    data: pd.DataFrame | np.ndarray,
    *,
    col: str = "x",
    dt: float = 1.0,
    smooth: int = 1,
) -> DescriptiveResult:
    """Estimate velocity, acceleration, and jerk from a position time series.

    Uses central finite differences for velocity and acceleration, with
    optional boxcar smoothing.

    Parameters
    ----------
    data : DataFrame or array
        Position values over time.
    col : str
        Column name if *data* is a DataFrame.
    dt : float
        Time step between samples.
    smooth : int
        Boxcar smoothing window (1 = no smoothing).

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with ``'velocity'``, ``'acceleration'``, ``'jerk'``.
    """
    x = _extract_col(data, col)
    if len(x) < 4:
        raise ValueError("Need at least 4 data points")
    if dt <= 0:
        raise ValueError("dt must be positive")
    vel = np.gradient(x, dt)
    acc = np.gradient(vel, dt)
    jerk = np.gradient(acc, dt)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        vel = np.convolve(vel, kernel, mode="same")
        acc = np.convolve(acc, kernel, mode="same")
        jerk = np.convolve(jerk, kernel, mode="same")
    return DescriptiveResult(
        name="Velocity profile",
        value={
            "velocity": vel.tolist(),
            "acceleration": acc.tolist(),
            "jerk": jerk.tolist(),
        },
        extra={
            "n": len(x),
            "dt": dt,
            "max_velocity": float(np.max(np.abs(vel))),
            "max_acceleration": float(np.max(np.abs(acc))),
            "max_jerk": float(np.max(np.abs(jerk))),
            "rms_velocity": float(np.sqrt(np.mean(vel**2))),
        },
    )


zoomv = velocity_profile


def cheatsheet() -> str:
    return "velocity_profile({}) -> Velocity profile estimation. 'You can't lock up the darkness"
