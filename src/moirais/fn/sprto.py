"""Spectral power ratio between two frequency bands."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spectral_ratio(
    x: np.ndarray,
    fs: float = 1.0,
    band1: tuple = (0.5, 4.0),
    band2: tuple = (4.0, 8.0),
) -> DescriptiveResult:
    """Compute spectral power ratio between two frequency bands.

    :param x: 1-D input signal.
    :param fs: Sampling frequency (default 1.0).
    :param band1: Lower frequency band (hz_low, hz_high).
    :param band2: Upper frequency band (hz_low, hz_high).
    :return: DescriptiveResult with power ratio value.
    """
    from moirais._spectral import periodogram, spectral_power_ratio

    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = periodogram(x, fs=fs)
    ratio = spectral_power_ratio(psd, freqs, band1, band2)
    return DescriptiveResult(name="spectral_power_ratio", value=float(ratio))


sprto = spectral_ratio


def cheatsheet() -> str:
    return "spectral_ratio({}) -> Spectral power ratio between two frequency bands."
