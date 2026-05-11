# morie.fn — function file (hadesllm/morie)
"""Savitzky-Golay filter. 'He who fights with monsters should be careful lest he thereby become a monster. — Friedrich Nietzsche'"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter as _savgol

from ._containers import DescriptiveResult


def savgol_smooth(
    y: np.ndarray,
    window: int = 11,
    polyorder: int = 3,
) -> DescriptiveResult:
    """
    Apply a Savitzky-Golay smoothing filter.

    Fits successive sub-sets of adjacent data points with a low-degree
    polynomial by the method of linear least squares.

    :param y: 1-D signal array.
    :param window: Length of the filter window (must be odd and > polyorder).
    :param polyorder: Order of the polynomial. Default 3.
    :return: DescriptiveResult with smoothed array in extra.
    :raises ValueError: If window or polyorder constraints violated.

    References
    ----------
    Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation
    of data by simplified least squares procedures. Analytical Chemistry,
    36(8), 1627--1639. doi:10.1021/ac60214a047
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < window:
        raise ValueError(f"y must be 1-D with length >= window ({window}).")
    if window % 2 == 0:
        raise ValueError("window must be odd.")
    if polyorder >= window:
        raise ValueError("polyorder must be < window.")

    smoothed = _savgol(y, window_length=window, polyorder=polyorder)
    residual = y - smoothed
    rmse = float(np.sqrt(np.mean(residual**2)))

    return DescriptiveResult(
        name="Savitzky-Golay Filter",
        value=rmse,
        extra={
            "smoothed": smoothed,
            "residuals": residual,
            "rmse": rmse,
            "window": window,
            "polyorder": polyorder,
            "n": len(y),
        },
    )


savgf = savgol_smooth


def cheatsheet() -> str:
    return "savgol_smooth({}) -> Savitzky-Golay filter. 'He who fights with monsters should be careful lest he thereby become a monster. — Friedrich Nietzsche' -- Y"
