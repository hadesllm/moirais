"""Spectral slope."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "It does not matter how slowly you go as long as you do not stop. — Confucius"


def spectral_slope(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute the spectral slope via linear regression on log magnitude.

    Fits a line to the log-magnitude spectrum and returns the slope,
    characterising the spectral tilt.

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
    mag = np.abs(X)
    mag = np.where(mag == 0, np.finfo(float).tiny, mag)
    log_mag = np.log10(mag)
    coeffs = np.polyfit(freqs, log_mag, 1)
    slope = float(coeffs[0])
    return DescriptiveResult(
        name="spectral_slope",
        value=slope,
        extra={"slope": slope, "intercept": float(coeffs[1]), "fs": fs},
    )


spslp = spectral_slope


def cheatsheet() -> str:
    return "spectral_slope({}) -> Spectral slope."
