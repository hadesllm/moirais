# morie.fn — function file (hadesllm/morie)
"""Cross-correlation function between two time series."""

import numpy as np


def ccf(
    x: np.ndarray,
    y: np.ndarray,
    nlags: int = 20,
    alpha: float = 0.05,
) -> dict:
    """
    Cross-correlation function between two time series.

    Computes the Pearson correlation between *x* and lagged *y*:

    .. math::

        \\hat{\\rho}_{xy}(h) = \\frac{\\sum_{t} (x_t - \\bar{x})(y_{t+h} - \\bar{y})}
            {\\sqrt{\\sum (x_t - \\bar{x})^2 \\sum (y_t - \\bar{y})^2}}

    for lags :math:`h = -\\text{nlags}, \\ldots, +\\text{nlags}`.

    :param x: 1-D array (reference series).
    :param y: 1-D array (lagged series). Same length as *x*.
    :param nlags: Maximum number of lags in each direction. Default 20.
    :param alpha: Significance level for the approximate CI
        (Bartlett white-noise bound). Default 0.05.
    :return: dict with ``lags`` (array), ``ccf_values`` (array),
        ``ci`` (float, half-width of white-noise CI).
    :raises ValueError: If *x* and *y* differ in length or are too short.

    References
    ----------
    Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. (2015). Time Series
    Analysis: Forecasting and Control (5th ed.). Wiley.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}.")
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 observations.")
    if nlags < 1:
        raise ValueError(f"nlags must be >= 1, got {nlags}.")
    nlags = min(nlags, n - 1)

    xm = x - np.mean(x)
    ym = y - np.mean(y)
    sx = np.sqrt(np.sum(xm**2))
    sy = np.sqrt(np.sum(ym**2))
    denom = sx * sy
    if denom == 0:
        raise ValueError("One or both series have zero variance.")

    lags_arr = np.arange(-nlags, nlags + 1)
    ccf_vals = np.zeros(len(lags_arr))

    for i, h in enumerate(lags_arr):
        if h >= 0:
            ccf_vals[i] = np.sum(xm[: n - h] * ym[h:]) / denom
        else:
            ccf_vals[i] = np.sum(xm[-h:] * ym[: n + h]) / denom

    z = __import__("scipy").stats.norm.ppf(1 - alpha / 2)
    ci = z / np.sqrt(n)

    return {
        "lags": lags_arr,
        "ccf_values": ccf_vals,
        "ci": float(ci),
    }


def cheatsheet() -> str:
    return "ccf({}) -> Cross-correlation function between two time series."
