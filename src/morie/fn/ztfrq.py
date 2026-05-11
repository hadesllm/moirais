"""Frequency response at specific frequencies."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Judge me by my size, do you?"


def freq_response_at(b, a, freqs, fs: float = 1.0) -> DescriptiveResult:
    """Evaluate H(f) at specific frequencies.

    Parameters
    ----------
    b : array-like
        Numerator coefficients.
    a : array-like
        Denominator coefficients.
    freqs : array-like
        Frequencies (Hz) at which to evaluate.
    fs : float
        Sampling frequency (Hz). Default 1.0.

    Returns
    -------
    DescriptiveResult
    """
    from scipy.signal import freqz

    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    worN = 2.0 * np.pi * freqs / fs
    w, h = freqz(b, a, worN=worN)
    mag = np.abs(h)
    phase = np.angle(h)
    return DescriptiveResult(
        name="freq_response_at",
        value=float(np.mean(mag)),
        extra={"frequencies": freqs, "magnitude": mag, "phase": phase, "H": h},
    )


ztfrq = freq_response_at


def cheatsheet() -> str:
    return "freq_response_at({}) -> Frequency response at specific frequencies."
