"""Wiener filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def wiener_filter(x, noise_psd=None, noise_fraction: float = 0.1) -> SignalResult:
    """Apply a Wiener filter to signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.
    noise_psd : array-like or None
        Noise power spectral density. If None, estimated from *x*.
    noise_fraction : float
        Fraction of signal used to estimate noise PSD when *noise_psd* is None.

    Returns
    -------
    SignalResult
    """
    from moirais._filters import wiener_filter as _wf

    x = np.asarray(x, dtype=float)
    result = _wf(x, noise_psd=noise_psd, noise_fraction=noise_fraction)
    return SignalResult(name="wiener_filter", filtered=result, fs=0.0, n_samples=len(x))


wienr = wiener_filter


def cheatsheet() -> str:
    return "wiener_filter({}) -> Wiener filter."
