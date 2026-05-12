# morie.fn -- function file (hadesllm/morie)
"""Exponential smoothing (Holt-Winters) for time series forecasting."""

import numpy as np


def ets(
    series: np.ndarray,
    trend: str | None = "add",
    seasonal: str | None = None,
    seasonal_periods: int = 12,
    forecast_periods: int = 12,
    alpha_init: float = 0.3,
    beta_init: float = 0.1,
    gamma_init: float = 0.1,
) -> dict:
    r"""
    Exponential smoothing (Holt-Winters method).

    Supports additive/multiplicative trend and seasonal components.

    .. math::

        \\ell_t = \\alpha (y_t - s_{t-m}) + (1 - \\alpha)(\\ell_{t-1} + b_{t-1})

        b_t = \\beta (\\ell_t - \\ell_{t-1}) + (1 - \\beta) b_{t-1}

        s_t = \\gamma (y_t - \\ell_t) + (1 - \\gamma) s_{t-m}

    (Additive form shown; multiplicative replaces subtraction with division.)

    :param series: 1-D array of time series values.
    :param trend: ``"add"``, ``"mul"``, or ``None``. Default ``"add"``.
    :param seasonal: ``"add"``, ``"mul"``, or ``None``. Default ``None``.
    :param seasonal_periods: Length of one seasonal cycle. Default 12.
    :param forecast_periods: Number of future periods to forecast. Default 12.
    :param alpha_init: Level smoothing parameter (0 < alpha < 1). Default 0.3.
    :param beta_init: Trend smoothing parameter. Default 0.1.
    :param gamma_init: Seasonal smoothing parameter. Default 0.1.
    :return: dict with ``fitted``, ``forecast``, ``params`` (alpha, beta,
        gamma), ``aic``, ``residuals``.
    :raises ValueError: On invalid trend/seasonal options or too-short series.

    References
    ----------
    Holt, C. C. (2004). Forecasting seasonals and trends by exponentially
    weighted moving averages. *International Journal of Forecasting*,
    20(1), 5-10. (Original 1957 ONR report.)

    Winters, P. R. (1960). Forecasting sales by exponentially weighted
    moving averages. *Management Science*, 6(3), 324-342.
    """
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n < 3:
        raise ValueError("Series must have at least 3 observations.")
    if trend not in ("add", "mul", None):
        raise ValueError(f"trend must be 'add', 'mul', or None, got {trend!r}.")
    if seasonal not in ("add", "mul", None):
        raise ValueError(f"seasonal must be 'add', 'mul', or None, got {seasonal!r}.")
    m = seasonal_periods
    if seasonal is not None and n < 2 * m:
        raise ValueError(f"Need at least 2*seasonal_periods={2 * m} observations, got {n}.")

    a = alpha_init
    b = beta_init
    g = gamma_init

    # Initialize level and trend
    level = np.zeros(n)
    tr = np.zeros(n)
    sea = np.zeros(n + m)
    fitted = np.zeros(n)

    if seasonal is not None:
        # Initialize seasonal from first cycle
        first_cycle = y[:m]
        cycle_mean = np.mean(first_cycle)
        if seasonal == "add":
            sea[:m] = first_cycle - cycle_mean
        else:
            sea[:m] = first_cycle / max(cycle_mean, 1e-12)
        level[0] = cycle_mean
    else:
        level[0] = y[0]

    if trend is not None:
        if n > 1:
            tr[0] = y[1] - y[0]
        else:
            tr[0] = 0.0

    fitted[0] = y[0]

    for t in range(1, n):
        y_t = y[t]
        l_prev = level[t - 1]
        b_prev = tr[t - 1]

        # De-seasonalize
        if seasonal == "add":
            s_prev = sea[t - m] if t >= m else sea[t % m]
            l_t = a * (y_t - s_prev) + (1 - a) * (l_prev + b_prev)
            b_t = b * (l_t - l_prev) + (1 - b) * b_prev if trend else 0.0
            s_t = g * (y_t - l_t) + (1 - g) * s_prev
            sea[t] = s_t
        elif seasonal == "mul":
            s_prev = sea[t - m] if t >= m else sea[t % m]
            s_prev = max(s_prev, 1e-12)
            l_t = a * (y_t / s_prev) + (1 - a) * (l_prev + b_prev)
            b_t = b * (l_t - l_prev) + (1 - b) * b_prev if trend else 0.0
            s_t = g * (y_t / max(l_t, 1e-12)) + (1 - g) * s_prev
            sea[t] = s_t
        else:
            l_t = a * y_t + (1 - a) * (l_prev + b_prev)
            b_t = b * (l_t - l_prev) + (1 - b) * b_prev if trend else 0.0

        if trend == "mul":
            b_t = b * (l_t / max(l_prev, 1e-12)) + (1 - b) * b_prev
            level[t] = l_t
            tr[t] = b_t
        else:
            level[t] = l_t
            tr[t] = b_t

        # Fitted value
        if seasonal == "add":
            fitted[t] = l_t + (b_t if trend else 0.0) + sea[t - m + 1] if t >= m else l_t + (b_t if trend else 0.0)
        elif seasonal == "mul":
            fitted[t] = (l_t + (b_t if trend else 0.0)) * (sea[t - m + 1] if t >= m else 1.0)
        else:
            fitted[t] = l_t + (b_t if trend else 0.0)

    # Forecast
    forecast = np.zeros(forecast_periods)
    l_last = level[n - 1]
    b_last = tr[n - 1]
    for h in range(forecast_periods):
        if trend == "add" or trend is None:
            f_h = l_last + (b_last * (h + 1) if trend else 0.0)
        else:
            f_h = l_last * (b_last ** (h + 1))
        if seasonal == "add":
            idx = (n + h) % m
            s_h = sea[n - m + idx] if (n - m + idx) >= 0 else 0.0
            f_h += s_h
        elif seasonal == "mul":
            idx = (n + h) % m
            s_h = sea[n - m + idx] if (n - m + idx) >= 0 else 1.0
            f_h *= s_h
        forecast[h] = f_h

    residuals = y - fitted
    sse = np.sum(residuals**2)
    n_params = 1 + (1 if trend else 0) + (1 if seasonal else 0) + 1  # +1 for sigma
    aic = n * np.log(sse / n) + 2 * n_params

    return {
        "fitted": fitted,
        "forecast": forecast,
        "params": {"alpha": a, "beta": b if trend else None, "gamma": g if seasonal else None},
        "aic": float(aic),
        "residuals": residuals,
    }


def cheatsheet() -> str:
    return "ets({}) -> Exponential smoothing (Holt-Winters) for time series forecas"
