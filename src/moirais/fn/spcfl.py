"""Spectral flatness (tonality measure)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spectral_flatness_fn(x: np.ndarray, fs: float = 1.0) -> DescriptiveResult:
    """Compute spectral flatness (Wiener entropy) of a signal.

    :param x: 1-D input signal.
    :param fs: Sampling frequency (default 1.0).
    :return: DescriptiveResult with flatness value in [0, 1].
    """
    from moirais._spectral import periodogram, spectral_flatness

    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = periodogram(x, fs=fs)
    flatness = spectral_flatness(psd)
    return DescriptiveResult(name="spectral_flatness", value=float(flatness))


spcfl = spectral_flatness_fn


def cheatsheet() -> str:
    return "spectral_flatness_fn({}) -> Spectral flatness (tonality measure)."
