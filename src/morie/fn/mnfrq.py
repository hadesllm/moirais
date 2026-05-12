# morie.fn -- function file (hadesllm/morie)
"""Mean frequency from spectral moments."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Who's the more foolish, the fool or the fool who follows him?"


def mean_frequency(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute the mean frequency from spectral moments.

    .. math::

        \\bar{f} = \\frac{m_1}{m_0}

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
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    psd = (np.abs(X) ** 2) / (N * fs)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    m0 = float(np.sum(psd * df))
    m1 = float(np.sum(freqs * psd * df))
    fmean = m1 / m0 if m0 > 0 else 0.0
    return DescriptiveResult(
        name="mean_frequency",
        value=fmean,
        extra={"mean_freq": fmean, "m0": m0, "m1": m1, "fs": fs},
    )


mnfrq = mean_frequency


def cheatsheet() -> str:
    return "mean_frequency({}) -> Mean frequency from spectral moments."
