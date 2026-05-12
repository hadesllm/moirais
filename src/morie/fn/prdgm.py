# morie.fn -- function file (hadesllm/morie)
"""Periodogram power spectral density estimate."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def periodogram_estimate(x: np.ndarray, fs: float = 1.0) -> DescriptiveResult:
    """Periodogram PSD estimate of a signal.

    :param x: 1-D input signal.
    :param fs: Sampling frequency (default 1.0).
    :return: DescriptiveResult with freqs and psd in extra.
    """
    from morie._spectral import periodogram

    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = periodogram(x, fs=fs)
    return DescriptiveResult(name="periodogram", value=None, extra={"freqs": freqs, "psd": psd})


prdgm = periodogram_estimate


def cheatsheet() -> str:
    return "periodogram_estimate({}) -> Periodogram power spectral density estimate."
