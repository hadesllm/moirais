"""Waste no more time arguing what a good person should be. Be one. -- Marcus Aurelius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def theta_method(y: np.ndarray, h: int = 12, alpha: float | None = None) -> DescriptiveResult:
    """
    Theta forecasting method (Assimakopoulos and Nikolopoulos, 2000).

    Decomposes the series into two theta-lines (theta=0 for linear trend,
    theta=2 for amplified curvature), forecasts each, and averages.

    :param y: 1-D time series.
    :type y: numpy.ndarray
    :param h: Forecast horizon. Default 12.
    :type h: int
    :param alpha: SES alpha for theta=2 line. If None, uses optimal.
    :type alpha: float or None
    :return: DescriptiveResult with forecasts.
    :rtype: DescriptiveResult

    References
    ----------
    Assimakopoulos V. & Nikolopoulos K. (2000). The theta model: a
    decomposition approach to forecasting. *International Journal of
    Forecasting*, 16(4), 521-530.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 4:
        raise ValueError(f"Need >= 4 observations, got {n}.")
    t_idx = np.arange(n, dtype=float)
    slope = (n * np.dot(t_idx, y) - t_idx.sum() * y.sum()) / (n * np.dot(t_idx, t_idx) - t_idx.sum() ** 2)
    intercept = (y.sum() - slope * t_idx.sum()) / n
    trend_line = intercept + slope * np.arange(n + h, dtype=float)
    theta2 = 2 * y - (intercept + slope * t_idx)
    if alpha is None:
        best_alpha = 0.5
        best_sse = np.inf
        for a_try in np.linspace(0.01, 0.99, 50):
            sm = np.empty(n)
            sm[0] = theta2[0]
            for t in range(1, n):
                sm[t] = a_try * theta2[t] + (1 - a_try) * sm[t - 1]
            sse = float(np.sum((theta2[1:] - sm[:-1]) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_alpha = a_try
        alpha = best_alpha
    ses = np.empty(n)
    ses[0] = theta2[0]
    for t in range(1, n):
        ses[t] = alpha * theta2[t] + (1 - alpha) * ses[t - 1]
    theta2_forecast = np.full(h, ses[-1])
    forecasts = 0.5 * (trend_line[n : n + h] + theta2_forecast)
    return DescriptiveResult(
        name="theta_method",
        value=float(forecasts[0]),
        extra={
            "forecasts": forecasts,
            "alpha": alpha,
            "slope": float(slope),
            "intercept": float(intercept),
        },
    )


thetf = theta_method


def cheatsheet() -> str:
    return "theta_method({}) -> Theta forecasting method. 'Your eyes can deceive you. Don't "
