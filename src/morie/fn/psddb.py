# morie.fn -- function file (hadesllm/morie)
"""Convert PSD to decibels."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def psd_decibels(x: np.ndarray, fs: float = 1.0) -> DescriptiveResult:
    """Convert PSD to decibels (10 * log10(psd)).

    :param x: 1-D input signal.
    :param fs: Sampling frequency (default 1.0).
    :return: DescriptiveResult with psd_db array in extra.
    """
    from morie._spectral import periodogram, psd_to_decibels

    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = periodogram(x, fs=fs)
    psd_db = psd_to_decibels(psd)
    return DescriptiveResult(name="psd_decibels", value=None, extra={"psd_db": psd_db, "freqs": freqs})


psddb = psd_decibels


def cheatsheet() -> str:
    return "psd_decibels({}) -> Convert PSD to decibels."
