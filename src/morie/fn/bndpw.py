# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Band power of a signal in a frequency range."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def band_power_fn(
    x: np.ndarray,
    fs: float = 1.0,
    f_low: float = 0.5,
    f_high: float = 4.0,
) -> DescriptiveResult:
    """Compute signal power in a specified frequency band.

    :param x: 1-D input signal.
    :param fs: Sampling frequency (default 1.0).
    :param f_low: Lower frequency bound in Hz.
    :param f_high: Upper frequency bound in Hz.
    :return: DescriptiveResult with band power value.
    """
    from morie._spectral import band_power, periodogram

    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = periodogram(x, fs=fs)
    power = band_power(psd, freqs, f_low, f_high)
    return DescriptiveResult(name="band_power", value=float(power))


bndpw = band_power_fn


def cheatsheet() -> str:
    return "band_power_fn({}) -> Band power of a signal in a frequency range."
