# moirais.fn — function file (hadesllm/moirais)
"""Double exponential smoothing (Holt's linear trend method)."""

import numpy as np

from ._containers import DescriptiveResult


def des(
    y: np.ndarray, alpha: float = 0.3, beta: float = 0.1, h: int = 1
) -> DescriptiveResult:
    """
    Double exponential smoothing (Holt's linear trend).

    .. math::

        \\ell_t &= \\alpha y_t + (1 - \\alpha)(\\ell_{t-1} + b_{t-1}) \\\\
        b_t &= \\beta (\\ell_t - \\ell_{t-1}) + (1 - \\beta) b_{t-1}

    :param y: 1-D time series.
    :param alpha: Level smoothing in (0, 1). Default 0.3.
    :param beta: Trend smoothing in (0, 1). Default 0.1.
    :param h: Forecast horizon. Default 1.
    :return: DescriptiveResult with fitted, level, trend, forecasts.
    :raises ValueError: If parameters out of range.

    References
    ----------
    Holt C.C. (1957/2004). Forecasting seasonals and trends by
    exponentially weighted moving averages. *IJF*, 20(1), 5-10.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}.")
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("alpha and beta must be in (0, 1).")
    level = np.zeros(n)
    trend = np.zeros(n)
    level[0] = y[0]
    trend[0] = y[1] - y[0] if n > 1 else 0
    fitted = np.zeros(n)
    fitted[0] = level[0] + trend[0]
    for t in range(1, n):
        level[t] = alpha * y[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        fitted[t] = level[t - 1] + trend[t - 1]
    forecasts = np.array([level[-1] + (i + 1) * trend[-1] for i in range(h)])
    return DescriptiveResult(
        name="des",
        value=float(forecasts[0]),
        extra={
            "fitted": fitted,
            "forecasts": forecasts,
            "level": level,
            "trend": trend,
            "alpha": alpha,
            "beta": beta,
            "n": n,
        },
    )


dexpn = des


def cheatsheet() -> str:
    return "des({}) -> Double exponential smoothing (Holt's linear trend)."
