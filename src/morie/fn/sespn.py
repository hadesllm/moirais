# morie.fn — function file (hadesllm/morie)
"""Simple exponential smoothing."""

import numpy as np

from ._containers import DescriptiveResult


def ses(y: np.ndarray, alpha: float = 0.3, h: int = 1) -> DescriptiveResult:
    r"""
    Simple exponential smoothing (SES).

    .. math::

        \\hat{y}_{t+1} = \\alpha y_t + (1 - \\alpha) \\hat{y}_t

    :param y: 1-D time series.
    :param alpha: Smoothing parameter in (0, 1). Default 0.3.
    :param h: Forecast horizon. Default 1.
    :return: DescriptiveResult with fitted values and forecast.
    :raises ValueError: If alpha not in (0,1) or series too short.

    References
    ----------
    Brown R.G. (1956). Exponential Smoothing for Predicting Demand.
    Cambridge, MA: Arthur D. Little.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    fitted = np.zeros(n)
    fitted[0] = y[0]
    for t in range(1, n):
        fitted[t] = alpha * y[t - 1] + (1 - alpha) * fitted[t - 1]
    level = alpha * y[-1] + (1 - alpha) * fitted[-1]
    forecasts = np.full(h, level)
    residuals = y - fitted
    return DescriptiveResult(
        name="ses",
        value=float(level),
        extra={
            "fitted": fitted,
            "forecasts": forecasts,
            "residuals": residuals,
            "alpha": alpha,
            "n": n,
        },
    )


sespn = ses


def cheatsheet() -> str:
    return "ses({}) -> Simple exponential smoothing."
