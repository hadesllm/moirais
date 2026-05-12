# morie.fn -- function file (hadesllm/morie)
"""It does not matter how slowly you go as long as you do not stop. -- Confucius"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _extract_col


def acceleration_profile(
    data: pd.DataFrame | np.ndarray,
    *,
    col: str = "x",
    dt: float = 1.0,
) -> DescriptiveResult:
    """Characterise the acceleration profile of a velocity time series.

    Computes acceleration phases (speeding up / slowing down / constant),
    peak acceleration, reaction time (time to first significant acceleration),
    and the acceleration-to-deceleration ratio.

    Parameters
    ----------
    data : DataFrame or array
        Velocity values over time.
    col : str
        Column name if *data* is a DataFrame.
    dt : float
        Time step between samples.

    Returns
    -------
    DescriptiveResult
        ``value`` = peak absolute acceleration.
    """
    v = _extract_col(data, col)
    if len(v) < 3:
        raise ValueError("Need at least 3 data points")
    if dt <= 0:
        raise ValueError("dt must be positive")
    acc = np.diff(v) / dt
    accel_mask = acc > 0
    decel_mask = acc < 0
    n_accel = int(accel_mask.sum())
    n_decel = int(decel_mask.sum())
    peak_acc = float(np.max(np.abs(acc)))
    threshold = 0.1 * peak_acc
    react_idx = np.where(np.abs(acc) > threshold)[0]
    reaction_time = float(react_idx[0] * dt) if len(react_idx) > 0 else float("inf")
    total_accel = float(np.sum(acc[accel_mask])) if n_accel > 0 else 0.0
    total_decel = float(np.abs(np.sum(acc[decel_mask]))) if n_decel > 0 else 0.0
    ratio = total_accel / total_decel if total_decel > 0 else float("inf")
    return DescriptiveResult(
        name="Acceleration profile",
        value=peak_acc,
        extra={
            "n": len(v),
            "dt": dt,
            "n_accelerating": n_accel,
            "n_decelerating": n_decel,
            "n_constant": len(acc) - n_accel - n_decel,
            "reaction_time": reaction_time,
            "accel_decel_ratio": round(ratio, 4),
            "rms_acceleration": float(np.sqrt(np.mean(acc**2))),
            "acceleration": acc.tolist(),
        },
    )


kflsh = acceleration_profile


def cheatsheet() -> str:
    return "acceleration_profile({}) -> Acceleration profile analysis. 'Fastest kid alive.' -- Kid F"
