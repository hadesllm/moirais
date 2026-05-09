"""Spectral edge frequency."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Study the past if you would define the future. — Confucius"


def spectral_edge_freq(x, fs: float = 1.0, pct: float = 0.95, **kwargs) -> DescriptiveResult:
    """Compute the spectral edge frequency.

    The frequency below which *pct* fraction of total power resides.

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.
    pct : float
        Fraction of total power (0 to 1).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    psd = np.abs(X) ** 2
    cumpower = np.cumsum(psd)
    threshold = cumpower[-1] * pct
    idx = int(np.searchsorted(cumpower, threshold))
    idx = min(idx, len(freqs) - 1)
    sef = float(freqs[idx])
    return DescriptiveResult(
        name="spectral_edge_freq",
        value=sef,
        extra={"sef": sef, "pct": pct, "fs": fs},
    )


spedf = spectral_edge_freq


def cheatsheet() -> str:
    return "spectral_edge_freq({}) -> Spectral edge frequency."
