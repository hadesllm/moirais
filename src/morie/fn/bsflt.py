# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bandstop Butterworth filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "In my experience, there is no such thing as luck."


def bandstop_filter(x, low, high, fs, order: int = 4) -> SignalResult:
    """Apply a bandstop Butterworth filter.

    Parameters
    ----------
    x : array-like
        Input signal.
    low : float
        Lower cutoff frequency (Hz).
    high : float
        Upper cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order. Default 4.

    Returns
    -------
    SignalResult
    """
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    sos = butter(order, [float(low) / nyq, float(high) / nyq], btype="bandstop", output="sos")
    y = sosfiltfilt(sos, x)
    return SignalResult(
        name="bandstop_filter",
        filtered=y,
        fs=fs,
        n_samples=len(x),
        extra={"low": low, "high": high, "order": order},
    )


bsflt = bandstop_filter


def cheatsheet() -> str:
    return "bandstop_filter({}) -> Bandstop Butterworth filter."
