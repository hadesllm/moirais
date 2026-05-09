"""SNR improvement estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def snr_improvement_fn(x_noisy, x_clean, x_filtered) -> DescriptiveResult:
    """Estimate the SNR improvement achieved by a filter.

    Parameters
    ----------
    x_noisy : array-like
        Noisy (unfiltered) signal.
    x_clean : array-like
        Clean reference signal.
    x_filtered : array-like
        Filtered signal output.

    Returns
    -------
    DescriptiveResult
        *value* is the SNR improvement in dB (filtered SNR minus input SNR).
    """
    from moirais._filters import snr_improvement as _snri

    x_noisy = np.asarray(x_noisy, dtype=float)
    x_clean = np.asarray(x_clean, dtype=float)
    x_filtered = np.asarray(x_filtered, dtype=float)
    improvement_db = _snri(x_noisy, x_clean, x_filtered)
    return DescriptiveResult(
        name="snr_improvement",
        value=improvement_db,
        extra={"unit": "dB"},
    )


snri = snr_improvement_fn


def cheatsheet() -> str:
    return "snr_improvement_fn({}) -> SNR improvement estimation."
