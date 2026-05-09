# moirais.fn — function file (hadesllm/moirais)
"""Notch (band-reject) filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Fear leads to anger. Anger leads to hate."


def notch_filter(x, freq, Q: float = 30.0, fs: float = 1.0) -> SignalResult:
    """Apply a notch (band-reject) filter at frequency *freq*.

    .. math::

        H(z) = \\frac{1 - 2\\cos(\\omega_0)z^{-1} + z^{-2}}
               {1 - 2r\\cos(\\omega_0)z^{-1} + r^2 z^{-2}}

    Parameters
    ----------
    x : array-like
        Input signal.
    freq : float
        Frequency to remove (Hz).
    Q : float
        Quality factor. Default 30.0.
    fs : float
        Sampling frequency (Hz). Default 1.0.

    Returns
    -------
    SignalResult
    """
    from scipy.signal import filtfilt, iirnotch

    x = np.asarray(x, dtype=float)
    w0 = float(freq) / (fs / 2.0)
    b, a = iirnotch(w0, Q)
    y = filtfilt(b, a, x)
    return SignalResult(
        name="notch_filter",
        filtered=y,
        fs=fs,
        n_samples=len(x),
        extra={"freq": freq, "Q": Q},
    )


ntchf = notch_filter


def cheatsheet() -> str:
    return "notch_filter({}) -> Notch (band-reject) filter."
