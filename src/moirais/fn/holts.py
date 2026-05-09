# moirais.fn — function file (hadesllm/moirais)
"""Holt's linear trend method. 'I love you.' 'Real knowledge is to know the extent of one's ignorance. — Confucius'

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def holts_method(y: np.ndarray, alpha: float = 0.3, beta: float = 0.1, h: int = 1) -> DescriptiveResult:
    """
    Holt's linear trend (double exponential smoothing).

    Updates level :math:`\\ell_t` and trend :math:`b_t`:

    .. math::

        \\ell_t &= \\alpha y_t + (1-\\alpha)(\\ell_{t-1} + b_{t-1}) \\\\
        b_t    &= \\beta(\\ell_t - \\ell_{t-1}) + (1-\\beta) b_{t-1}

    :param y: 1-D time series.
    :type y: numpy.ndarray
    :param alpha: Level smoothing in (0, 1). Default 0.3.
    :type alpha: float
    :param beta: Trend smoothing in (0, 1). Default 0.1.
    :type beta: float
    :param h: Forecast horizon. Default 1.
    :type h: int
    :return: DescriptiveResult with fitted values and h-step forecast.
    :rtype: DescriptiveResult

    References
    ----------
    Holt C.C. (1957). Forecasting seasonals and trends by exponentially
    weighted moving averages. ONR Research Memorandum No. 52.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 3:
        raise ValueError(f"Need >= 3 observations, got {n}.")
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("alpha and beta must be in (0, 1).")
    level = np.empty(n)
    trend = np.empty(n)
    level[0] = y[0]
    trend[0] = y[1] - y[0]
    fitted = np.empty(n)
    fitted[0] = level[0]
    for t in range(1, n):
        level[t] = alpha * y[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        fitted[t] = level[t - 1] + trend[t - 1]
    forecasts = np.array([level[-1] + (i + 1) * trend[-1] for i in range(h)])
    return DescriptiveResult(
        name="holts_method",
        value=float(forecasts[0]),
        extra={
            "fitted": fitted,
            "forecasts": forecasts,
            "level": level,
            "trend": trend,
            "alpha": alpha,
            "beta": beta,
        },
    )


holts = holts_method


def cheatsheet() -> str:
    return "holts_method({}) -> Holt's linear trend method. 'I love you.' 'Real knowledge is to know the extent of one's ignorance. — Confucius'
