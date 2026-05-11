"""Savitzky-Golay smoothing filter."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ._containers import SignalResult


def savgol_smooth(
    x: np.ndarray,
    *,
    window: int = 11,
    polyorder: int = 3,
) -> SignalResult:
    """Savitzky-Golay polynomial smoothing.

    :param x: 1-D input signal.
    :param window: Window length (must be odd, default 11).
    :param polyorder: Polynomial order (default 3).
    :return: SignalResult with smoothed signal.
    """
    x = np.asarray(x, dtype=float).ravel()
    if window % 2 == 0:
        window += 1
    y = savgol_filter(x, window, polyorder)
    return SignalResult(
        name="savgol_smooth",
        filtered=y,
        fs=0.0,
        n_samples=len(y),
        extra={"window": window, "polyorder": polyorder},
    )


sgolay = savgol_smooth


def cheatsheet() -> str:
    return "savgol_smooth({}) -> Savitzky-Golay smoothing filter."
