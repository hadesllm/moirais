"""Spectral entropy."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There is always a bigger fish."


def spectral_entropy(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    r"""Compute the spectral entropy of signal *x*.

    .. math::

        H = -\\sum p(f) \\cdot \\log_2(p(f))

    where *p(f)* is the normalized power spectral density.

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
    X = np.fft.rfft(x)
    psd = np.abs(X) ** 2
    total = np.sum(psd)
    if total == 0:
        return DescriptiveResult(name="spectral_entropy", value=0.0, extra={"entropy": 0.0})
    p = psd / total
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))
    return DescriptiveResult(
        name="spectral_entropy",
        value=entropy,
        extra={"entropy": entropy, "fs": fs},
    )


spent = spectral_entropy


def cheatsheet() -> str:
    return "spectral_entropy({}) -> Spectral entropy."
