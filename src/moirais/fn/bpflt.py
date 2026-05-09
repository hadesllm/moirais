# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bandpass Butterworth filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "No man ever steps in the same river twice. — Heraclitus"


def bandpass_filter(x, low, high, fs, order: int = 4) -> SignalResult:
    """Apply a bandpass Butterworth filter.

    .. math::

        H_{\\text{bp}}(\\omega) = H_{\\text{lp}}(\\omega - \\omega_L) \\cdot
                                  H_{\\text{hp}}(\\omega - \\omega_H)

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
    sos = butter(order, [float(low) / nyq, float(high) / nyq], btype="bandpass", output="sos")
    y = sosfiltfilt(sos, x)
    return SignalResult(
        name="bandpass_filter",
        filtered=y,
        fs=fs,
        n_samples=len(x),
        extra={"low": low, "high": high, "order": order},
    )


bpflt = bandpass_filter


def cheatsheet() -> str:
    return "bandpass_filter({}) -> Bandpass Butterworth filter."
