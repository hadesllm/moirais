# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Butterworth bandpass filter."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ._containers import SignalResult


def butter_bandpass(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    *,
    order: int = 4,
) -> SignalResult:
    """Zero-phase Butterworth bandpass filter.

    :param x: 1-D input signal.
    :param fs: Sampling frequency in Hz.
    :param low: Lower cutoff frequency in Hz.
    :param high: Upper cutoff frequency in Hz.
    :param order: Filter order (default 4).
    :return: SignalResult with filtered signal.
    """
    x = np.asarray(x, dtype=float).ravel()
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    return SignalResult(
        name="butter_bandpass",
        filtered=y,
        fs=fs,
        n_samples=len(y),
        extra={"low": low, "high": high, "order": order},
    )


buttbp = butter_bandpass


def cheatsheet() -> str:
    return "butter_bandpass({}) -> Butterworth bandpass filter."
