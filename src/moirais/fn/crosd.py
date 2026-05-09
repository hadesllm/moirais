# moirais.fn — function file (hadesllm/moirais)
"""Croston's method for intermittent demand forecasting."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ["crosd", "croston_method"]


def croston_method(
    y,
    alpha: float = 0.1,
    h: int = 1,
    sba: bool = False,
) -> DescriptiveResult:
    """Croston's method for intermittent / sparse time series forecasting.

    Separately applies single exponential smoothing to demand sizes and to
    inter-demand intervals, then combines them to produce a per-period demand
    forecast.

    Algorithm (Croston 1972):
    - At each demand occurrence (y[t] > 0):
      ``a[t] = alpha * y[t] + (1 - alpha) * a[t-1]``   (demand level)
      ``p[t] = alpha * q[t]  + (1 - alpha) * p[t-1]``  (interval level)
      where ``q[t]`` is the observed inter-arrival time.
    - Forecast: ``f = a / p``
    - Syntetos-Boylan Approximation (sba=True): ``f_sba = (1 - alpha/2) * a/p``

    Parameters
    ----------
    y : array-like
        Non-negative time series with many zeros (n,).
    alpha : float
        Exponential smoothing parameter in (0, 1).  Default 0.1.
    h : int
        Steps ahead for the forecast (currently only h=1 is produced).
        Default 1.
    sba : bool
        If True apply the Syntetos-Boylan bias correction.  Default False.

    Returns
    -------
    DescriptiveResult
        value: float — per-period demand forecast (f or f_sba).
        extra keys:
          'demand_level'   : float — final smoothed demand size (a).
          'interval_level' : float — final smoothed inter-demand interval (p).
          'combined_forecast': float — a / p.
          'sba_forecast'   : float — (1 - alpha/2) * a / p.
          'n_demands'      : int — number of non-zero observations.
          'alpha'          : float.

    Raises
    ------
    ValueError
        If series too short or alpha not in (0, 1).

    References
    ----------
    Croston J.D. (1972). Forecasting and stock control for intermittent
    demands. Operational Research Quarterly, 23(3), 289-303.

    Syntetos A.A. & Boylan J.E. (2001). On the bias of intermittent demand
    estimates. International Journal of Production Economics, 71(1), 457-466.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 2:
        raise ValueError(f"Need >= 2 observations, got {n}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    nonzero_idx = np.flatnonzero(y > 0.0)
    n_demands = len(nonzero_idx)

    if n_demands == 0:
        return DescriptiveResult(
            name="croston_method",
            value=0.0,
            extra={
                "demand_level": 0.0,
                "interval_level": float(n),
                "combined_forecast": 0.0,
                "sba_forecast": 0.0,
                "n_demands": 0,
                "alpha": float(alpha),
            },
        )

    # Initialise with the first demand observation.
    a = float(y[nonzero_idx[0]])   # smoothed demand size
    q_prev_idx = nonzero_idx[0]
    # Initial interval: distance from time 0 to first demand (at least 1).
    p = float(nonzero_idx[0] + 1)

    # Update at each subsequent demand occurrence.
    for idx in nonzero_idx[1:]:
        q = float(idx - q_prev_idx)   # observed inter-arrival time
        a = alpha * float(y[idx]) + (1.0 - alpha) * a
        p = alpha * q + (1.0 - alpha) * p
        q_prev_idx = idx

    combined = a / p if p > 0.0 else 0.0
    sba_forecast = (1.0 - alpha / 2.0) * combined

    forecast = sba_forecast if sba else combined

    return DescriptiveResult(
        name="croston_method",
        value=float(forecast),
        extra={
            "demand_level": float(a),
            "interval_level": float(p),
            "combined_forecast": float(combined),
            "sba_forecast": float(sba_forecast),
            "n_demands": int(n_demands),
            "alpha": float(alpha),
        },
    )


crosd = croston_method


def cheatsheet() -> str:
    return "croston_method(y, alpha=0.1, sba=False) -> Croston's intermittent demand forecast."
