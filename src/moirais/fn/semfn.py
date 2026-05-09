# moirais.fn — function file (hadesllm/moirais)
"""Spectral error measure between two signals."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spectral_error_fn(x1: np.ndarray, x2: np.ndarray, fs: float = 1.0) -> DescriptiveResult:
    """Compute spectral error measure (log-spectral distance) between two signals.

    :param x1: First 1-D input signal.
    :param x2: Second 1-D input signal (same length as x1).
    :param fs: Sampling frequency (default 1.0).
    :return: DescriptiveResult with spectral error measure value.
    """
    from moirais._adaptive import spectral_error_measure
    from moirais._spectral import periodogram

    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    _, psd1 = periodogram(x1, fs=fs)
    _, psd2 = periodogram(x2, fs=fs)
    sem = spectral_error_measure(psd1, psd2)
    return DescriptiveResult(name="spectral_error_measure", value=float(sem))


semfn = spectral_error_fn


def cheatsheet() -> str:
    return "spectral_error_fn({}) -> Spectral error measure between two signals."
