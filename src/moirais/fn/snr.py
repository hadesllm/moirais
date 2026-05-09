"""SNR estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def snr_estimate_fn(signal, noise) -> DescriptiveResult:
    """Estimate the signal-to-noise ratio (SNR) in decibels.

    Parameters
    ----------
    signal : array-like
        Clean signal array.
    noise : array-like
        Noise array (same length as *signal*).

    Returns
    -------
    DescriptiveResult
        *value* is the SNR in dB.
    """
    from moirais._filters import snr_estimate as _snr

    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    snr_db = _snr(signal, noise)
    return DescriptiveResult(name="snr_estimate", value=snr_db, extra={"unit": "dB"})


snr = snr_estimate_fn


def cheatsheet() -> str:
    return "snr_estimate_fn({}) -> SNR estimation."
