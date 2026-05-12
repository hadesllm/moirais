# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Butterworth highpass filter."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ._containers import SignalResult


def butter_highpass(
    x: np.ndarray,
    fs: float,
    cutoff: float,
    *,
    order: int = 4,
) -> SignalResult:
    """Zero-phase Butterworth highpass filter.

    :param x: 1-D input signal.
    :param fs: Sampling frequency in Hz.
    :param cutoff: Cutoff frequency in Hz.
    :param order: Filter order (default 4).
    :return: SignalResult with filtered signal.
    """
    x = np.asarray(x, dtype=float).ravel()
    sos = butter(order, cutoff, btype="high", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    return SignalResult(
        name="butter_highpass",
        filtered=y,
        fs=fs,
        n_samples=len(y),
        extra={"cutoff": cutoff, "order": order},
    )


butthp = butter_highpass


def cheatsheet() -> str:
    return "butter_highpass({}) -> Butterworth highpass filter."
