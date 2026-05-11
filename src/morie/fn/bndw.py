# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Spectral bandwidth."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The strongest stars have hearts of kyber."


def bandwidth_compute(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute the spectral bandwidth.

    .. math::

        BW = \\sqrt{\\frac{m_2}{m_0} - \\left(\\frac{m_1}{m_0}\\right)^2}

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
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    psd = (np.abs(X) ** 2) / (N * fs)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    m0 = float(np.sum(psd * df))
    m1 = float(np.sum(freqs * psd * df))
    m2 = float(np.sum(freqs**2 * psd * df))
    if m0 > 0:
        bw = float(np.sqrt(m2 / m0 - (m1 / m0) ** 2))
    else:
        bw = 0.0
    return DescriptiveResult(
        name="bandwidth_compute",
        value=bw,
        extra={"bandwidth": bw, "m0": m0, "m1": m1, "m2": m2, "fs": fs},
    )


bndw = bandwidth_compute


def cheatsheet() -> str:
    return "bandwidth_compute({}) -> Spectral bandwidth."
