# moirais.fn — function file (hadesllm/moirais)
"""Simple exponential smoothing. 'Much to learn, you still have.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def exponential_smooth(y: np.ndarray, alpha: float = 0.3) -> DescriptiveResult:
    """
    Simple Exponential Smoothing (SES).

    Produces one-step-ahead forecasts using the recursion:

    .. math::

        \\hat{y}_{t+1} = \\alpha y_t + (1 - \\alpha) \\hat{y}_t

    :param y: 1-D time series of observations.
    :type y: numpy.ndarray
    :param alpha: Smoothing parameter in (0, 1). Default 0.3.
    :type alpha: float
    :return: DescriptiveResult with smoothed series and final forecast.
    :rtype: DescriptiveResult
    :raises ValueError: If alpha not in (0, 1).

    References
    ----------
    Brown R.G. (1956). *Exponential Smoothing for Predicting Demand*.
    Cambridge, MA: Arthur D. Little Inc.
    """
    y = np.asarray(y, dtype=float).ravel()
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    n = len(y)
    if n < 2:
        raise ValueError(f"Need >= 2 observations, got {n}.")
    smoothed = np.empty(n)
    smoothed[0] = y[0]
    for t in range(1, n):
        smoothed[t] = alpha * y[t] + (1 - alpha) * smoothed[t - 1]
    forecast = float(smoothed[-1])
    return DescriptiveResult(
        name="exponential_smoothing",
        value=forecast,
        extra={"smoothed": smoothed, "alpha": alpha, "forecast": forecast},
    )


expsm = exponential_smooth


def cheatsheet() -> str:
    return "exponential_smooth({}) -> Simple exponential smoothing. 'Much to learn, you still have"
