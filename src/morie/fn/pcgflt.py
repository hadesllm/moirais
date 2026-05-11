# morie.fn — function file (hadesllm/morie)
"""PCG bandpass preprocessing filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def pcg_filter(
    pcg: np.ndarray,
    fs: float,
    *,
    low: float = 25.0,
    high: float = 400.0,
    order: int = 4,
) -> SignalResult:
    """Bandpass filter for phonocardiogram preprocessing.

    :param pcg: 1-D PCG signal.
    :param fs: Sampling frequency in Hz.
    :param low: Lower cutoff in Hz (default 25).
    :param high: Upper cutoff in Hz (default 400).
    :param order: Filter order (default 4).
    :return: SignalResult with filtered signal.
    """
    from .buttbp import butter_bandpass

    result = butter_bandpass(np.asarray(pcg, dtype=float).ravel(), fs, low, high, order=order)
    return SignalResult(
        name="pcg_filter",
        filtered=result.filtered,
        fs=fs,
        n_samples=len(result.filtered),
        extra={"low": low, "high": high, "order": order},
    )


pcgflt = pcg_filter


def cheatsheet() -> str:
    return "pcg_filter({}) -> PCG bandpass preprocessing filter."
