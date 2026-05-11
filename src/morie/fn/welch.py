"""Welch power spectral density estimation."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch as _welch

from ._containers import SignalResult


def welch_psd(
    x: np.ndarray,
    fs: float,
    *,
    nperseg: int = 256,
) -> SignalResult:
    """Welch's method for power spectral density estimation.

    :param x: 1-D input signal.
    :param fs: Sampling frequency in Hz.
    :param nperseg: Length of each segment (default 256).
    :return: SignalResult with PSD in ``filtered`` and frequencies in ``extra["freqs"]``.
    """
    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = _welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    return SignalResult(
        name="welch_psd",
        filtered=psd,
        fs=fs,
        n_samples=len(psd),
        extra={"freqs": freqs},
    )


welch = welch_psd


def cheatsheet() -> str:
    return "welch_psd({}) -> Welch power spectral density estimation."
