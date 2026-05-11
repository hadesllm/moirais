"""Spectral centroid."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Strike me down, and I will become more powerful than you imagine."


def spectral_centroid(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute the spectral centroid.

    .. math::

        C = \\frac{\\sum f \\cdot |X(f)|^2}{\\sum |X(f)|^2}

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    mag2 = np.abs(X) ** 2
    total = float(np.sum(mag2))
    if total == 0:
        centroid = 0.0
    else:
        centroid = float(np.sum(freqs * mag2) / total)
    return DescriptiveResult(
        name="spectral_centroid",
        value=centroid,
        extra={"centroid": centroid, "fs": fs},
    )


spcnt = spectral_centroid


def cheatsheet() -> str:
    return "spectral_centroid({}) -> Spectral centroid."
