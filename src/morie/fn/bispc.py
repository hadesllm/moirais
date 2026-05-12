# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bispectrum estimation for nonlinear coupling detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bispectrum_fn(
    x: np.ndarray,
    fs: float = 1.0,
    nfft: int = 256,
) -> DescriptiveResult:
    """Compute bispectrum of a signal.

    :param x: 1-D input signal.
    :param fs: Sampling frequency in Hz (default 1.0).
    :param nfft: FFT length (default 256).
    :return: DescriptiveResult with bispectrum matrix and frequencies.
    """
    from morie._adaptive import bispectrum

    x = np.asarray(x, dtype=float).ravel()
    bispec, freqs = bispectrum(x, fs=fs, nfft=nfft)
    return DescriptiveResult(
        name="bispectrum",
        value=len(freqs),
        extra={"bispectrum": bispec, "frequencies": freqs},
    )


bispc = bispectrum_fn


def cheatsheet() -> str:
    return "bispectrum_fn({}) -> Bispectrum estimation for nonlinear coupling detection."
