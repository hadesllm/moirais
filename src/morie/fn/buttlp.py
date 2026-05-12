# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Butterworth lowpass filter."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ._containers import SignalResult


def butter_lowpass(
    x: np.ndarray,
    fs: float,
    cutoff: float,
    *,
    order: int = 4,
) -> SignalResult:
    """Zero-phase Butterworth lowpass filter.

    :param x: 1-D input signal.
    :param fs: Sampling frequency in Hz.
    :param cutoff: Cutoff frequency in Hz.
    :param order: Filter order (default 4).
    :return: SignalResult with filtered signal.
    """
    x = np.asarray(x, dtype=float).ravel()
    sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    return SignalResult(
        name="butter_lowpass",
        filtered=y,
        fs=fs,
        n_samples=len(y),
        extra={"cutoff": cutoff, "order": order},
    )


buttlp = butter_lowpass


def cheatsheet() -> str:
    return "butter_lowpass({}) -> Butterworth lowpass filter."
