# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Butterworth bandstop (notch) filter."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ._containers import SignalResult


def butter_bandstop(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    *,
    order: int = 4,
) -> SignalResult:
    """Zero-phase Butterworth bandstop filter.

    :param x: 1-D input signal.
    :param fs: Sampling frequency in Hz.
    :param low: Lower cutoff frequency in Hz.
    :param high: Upper cutoff frequency in Hz.
    :param order: Filter order (default 4).
    :return: SignalResult with filtered signal.
    """
    x = np.asarray(x, dtype=float).ravel()
    sos = butter(order, [low, high], btype="bandstop", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    return SignalResult(
        name="butter_bandstop",
        filtered=y,
        fs=fs,
        n_samples=len(y),
        extra={"low": low, "high": high, "order": order},
    )


buttbs = butter_bandstop


def cheatsheet() -> str:
    return "butter_bandstop({}) -> Butterworth bandstop (notch) filter."
