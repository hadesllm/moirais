"""Spectral entropy of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spectral_entropy_fn(x: np.ndarray, fs: float = 1.0) -> DescriptiveResult:
    """Compute normalized spectral entropy of a signal.

    :param x: 1-D input signal.
    :param fs: Sampling frequency (default 1.0).
    :return: DescriptiveResult with entropy value in [0, 1].
    """
    from moirais._spectral import periodogram, spectral_entropy

    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = periodogram(x, fs=fs)
    entropy = spectral_entropy(psd)
    return DescriptiveResult(name="spectral_entropy", value=float(entropy))


spcen = spectral_entropy_fn


def cheatsheet() -> str:
    return "spectral_entropy_fn({}) -> Spectral entropy of a signal."
