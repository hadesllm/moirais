# morie.fn -- function file (hadesllm/morie)
"""Spectral edge frequency of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spectral_edge_freq(x: np.ndarray, fs: float = 1.0, pct: float = 0.95) -> DescriptiveResult:
    """Compute spectral edge frequency at given power percentile.

    :param x: 1-D input signal.
    :param fs: Sampling frequency (default 1.0).
    :param pct: Power percentile threshold (default 0.95).
    :return: DescriptiveResult with spectral edge frequency value.
    """
    from morie._spectral import periodogram, spectral_edge_frequency

    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = periodogram(x, fs=fs)
    freq = spectral_edge_frequency(psd, freqs, pct=pct)
    return DescriptiveResult(name="spectral_edge_frequency", value=float(freq))


sefrq = spectral_edge_freq


def cheatsheet() -> str:
    return "spectral_edge_freq({}) -> Spectral edge frequency of a signal."
