# morie.fn -- function file (hadesllm/morie)
"""All models are wrong, but some are useful. -- George E. P. Box"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def naive_forecast(y: np.ndarray, h: int = 1, method: str = "last", season: int = 1) -> DescriptiveResult:
    """
    Naive forecasting methods.

    - ``"last"``: forecast = last observed value (random walk).
    - ``"mean"``: forecast = series mean.
    - ``"seasonal"``: forecast repeats the last seasonal cycle.
    - ``"drift"``: forecast = last value + average change per period.

    :param y: 1-D time series.
    :type y: numpy.ndarray
    :param h: Forecast horizon. Default 1.
    :type h: int
    :param method: One of ``"last"``, ``"mean"``, ``"seasonal"``, ``"drift"``.
    :type method: str
    :param season: Seasonal period for ``"seasonal"`` method. Default 1.
    :type season: int
    :return: DescriptiveResult with forecast array.
    :rtype: DescriptiveResult

    References
    ----------
    Hyndman R.J. & Athanasopoulos G. (2021). *Forecasting: Principles
    and Practice*, 3rd ed. OTexts. Chapter 5.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 1:
        raise ValueError("Need at least 1 observation.")
    method = method.lower()
    if method == "last":
        forecasts = np.full(h, y[-1])
    elif method == "mean":
        forecasts = np.full(h, np.mean(y))
    elif method == "seasonal":
        if season < 1 or season > n:
            raise ValueError(f"season must be in [1, {n}], got {season}.")
        last_cycle = y[-season:]
        forecasts = np.array([last_cycle[i % season] for i in range(h)])
    elif method == "drift":
        if n < 2:
            raise ValueError("Drift needs >= 2 observations.")
        drift = (y[-1] - y[0]) / (n - 1)
        forecasts = np.array([y[-1] + (i + 1) * drift for i in range(h)])
    else:
        raise ValueError(f"Unknown method '{method}'.")
    return DescriptiveResult(
        name="naive_forecast",
        value=float(forecasts[0]),
        extra={"forecasts": forecasts, "method": method, "h": h},
    )


naive = naive_forecast


def cheatsheet() -> str:
    return "All models are wrong, but some are useful. -- George E. P. Box"
