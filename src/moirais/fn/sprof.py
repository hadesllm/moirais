"""Spectral rolloff frequency."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Many of the truths we cling to depend on our point of view."


def spectral_rolloff(x, fs: float = 1.0, pct: float = 0.85, **kwargs) -> DescriptiveResult:
    """Compute the spectral rolloff frequency.

    The frequency below which *pct* fraction of the total spectral
    energy is contained.

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.
    pct : float
        Rolloff percentage (0 to 1).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    mag = np.abs(X) ** 2
    cumsum = np.cumsum(mag)
    threshold = cumsum[-1] * pct
    idx = int(np.searchsorted(cumsum, threshold))
    idx = min(idx, len(freqs) - 1)
    rolloff = float(freqs[idx])
    return DescriptiveResult(
        name="spectral_rolloff",
        value=rolloff,
        extra={"rolloff": rolloff, "pct": pct, "fs": fs},
    )


sprof = spectral_rolloff


def cheatsheet() -> str:
    return "spectral_rolloff({}) -> Spectral rolloff frequency."
