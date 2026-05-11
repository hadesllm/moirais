"""Spectral moment of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spectral_moment_fn(x: np.ndarray, fs: float = 1.0, order: int = 0) -> DescriptiveResult:
    """Compute spectral moment of order n from signal x.

    :param x: 1-D input signal.
    :param fs: Sampling frequency (default 1.0).
    :param order: Moment order (default 0).
    :return: DescriptiveResult with moment value.
    """
    from morie._spectral import periodogram, spectral_moment

    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = periodogram(x, fs=fs)
    moment = spectral_moment(psd, freqs, order=order)
    return DescriptiveResult(name="spectral_moment", value=float(moment))


spmom = spectral_moment_fn


def cheatsheet() -> str:
    return "spectral_moment_fn({}) -> Spectral moment of a signal."
