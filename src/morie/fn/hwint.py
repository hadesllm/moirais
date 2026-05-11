# morie.fn — function file (hadesllm/morie)
"""Holt-Winters additive seasonal method. 'The belonging you seek is ahead of you.' -- Maz Kanata"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def holt_winters(
    y: np.ndarray, alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1, season: int = 12, h: int = 1
) -> DescriptiveResult:
    """
    Holt-Winters additive seasonal smoothing.

    Updates level, trend, and seasonal components:

    .. math::

        \\ell_t &= \\alpha(y_t - s_{t-m}) + (1-\\alpha)(\\ell_{t-1} + b_{t-1}) \\\\
        b_t &= \\beta(\\ell_t - \\ell_{t-1}) + (1-\\beta) b_{t-1} \\\\
        s_t &= \\gamma(y_t - \\ell_t) + (1-\\gamma) s_{t-m}

    :param y: 1-D time series (length >= 2 * season).
    :type y: numpy.ndarray
    :param alpha: Level smoothing in (0, 1). Default 0.3.
    :param beta: Trend smoothing in (0, 1). Default 0.1.
    :param gamma: Seasonal smoothing in (0, 1). Default 0.1.
    :param season: Seasonal period (e.g. 12 for monthly). Default 12.
    :param h: Forecast horizon. Default 1.
    :return: DescriptiveResult with fitted, forecasts, components.
    :rtype: DescriptiveResult

    References
    ----------
    Winters P.R. (1960). Forecasting sales by exponentially weighted
    moving averages. *Management Science*, 6(3), 324-342.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    m = season
    if n < 2 * m:
        raise ValueError(f"Need >= {2 * m} observations for season={m}, got {n}.")
    if not (0 < alpha < 1 and 0 < beta < 1 and 0 < gamma < 1):
        raise ValueError("alpha, beta, gamma must be in (0, 1).")
    seasonal = np.zeros(n + h)
    first_season_mean = np.mean(y[:m])
    for i in range(m):
        seasonal[i] = y[i] - first_season_mean
    level = np.zeros(n)
    trend = np.zeros(n)
    level[0] = first_season_mean
    trend[0] = (np.mean(y[m : 2 * m]) - np.mean(y[:m])) / m
    fitted = np.zeros(n)
    fitted[0] = level[0] + trend[0] + seasonal[0]
    for t in range(1, n):
        level[t] = alpha * (y[t] - seasonal[t - m]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        seasonal[t] = gamma * (y[t] - level[t]) + (1 - gamma) * seasonal[t - m]
        fitted[t] = level[t - 1] + trend[t - 1] + seasonal[t - m]
    forecasts = np.array([level[-1] + (i + 1) * trend[-1] + seasonal[n - m + ((i) % m)] for i in range(h)])
    return DescriptiveResult(
        name="holt_winters",
        value=float(forecasts[0]),
        extra={
            "fitted": fitted,
            "forecasts": forecasts,
            "level": level,
            "trend": trend,
            "seasonal": seasonal[:n],
            "season": m,
        },
    )


hwint = holt_winters


def cheatsheet() -> str:
    return "holt_winters({}) -> Holt-Winters additive seasonal method. 'The belonging you se"
