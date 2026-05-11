# morie.fn — function file (hadesllm/morie)
"""Holt-Winters exponential smoothing."""

import numpy as np

from ._containers import DescriptiveResult


def exponential_smooth(
    y: np.ndarray, alpha: float = 0.2, beta: float | None = None, gamma: float | None = None, period: int = 12
) -> DescriptiveResult:
    """
    Holt-Winters exponential smoothing (additive seasonality).

    :param y: (n,) time series.
    :param alpha: Level smoothing parameter (0, 1).
    :param beta: Trend smoothing parameter (None = no trend).
    :param gamma: Seasonal smoothing parameter (None = no seasonality).
    :param period: Seasonal period.
    :return: DescriptiveResult with smoothed values and components.
    :raises ValueError: If alpha not in (0, 1) or series too short.

    References
    ----------
    Hyndman RJ et al. (2008). Forecasting with Exponential Smoothing.
    Springer.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    if n < 4:
        raise ValueError("Need at least 4 observations.")
    level = np.zeros(n)
    trend = np.zeros(n)
    seasonal = np.zeros(n)
    smoothed = np.zeros(n)
    level[0] = y[0]
    if beta is not None and n > 1:
        trend[0] = y[1] - y[0]
    if gamma is not None and n >= period:
        for i in range(period):
            seasonal[i] = y[i] - np.mean(y[:period])
    for t in range(1, n):
        s_prev = seasonal[t - period] if (gamma is not None and t >= period) else 0.0
        level[t] = alpha * (y[t] - s_prev) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        if beta is not None:
            trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        if gamma is not None and t >= period:
            seasonal[t] = gamma * (y[t] - level[t]) + (1 - gamma) * seasonal[t - period]
        smoothed[t] = level[t] + trend[t] + (seasonal[t] if gamma else 0.0)
    smoothed[0] = y[0]
    residuals = y - smoothed
    mse = float(np.mean(residuals[1:] ** 2))
    return DescriptiveResult(
        name="exponential_smoothing",
        value=mse,
        extra={
            "smoothed": smoothed,
            "level": level,
            "trend": trend,
            "seasonal": seasonal,
            "residuals": residuals,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "period": period,
            "mse": mse,
            "n": n,
        },
    )


expns = exponential_smooth


def cheatsheet() -> str:
    return "exponential_smooth({}) -> Holt-Winters exponential smoothing."
