# moirais.fn — function file (hadesllm/moirais)
"""Power spectral density via periodogram."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I've got a bad feeling about this."


def periodogram(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute the power spectral density via the periodogram method.

    .. math::

        S(f) = \\frac{1}{N \\cdot f_s} |X(f)|^2

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
    return DescriptiveResult(
        name="periodogram",
        value=float(np.max(psd)),
        extra={"psd": psd, "freqs": freqs, "fs": fs, "N": N},
    )


psdpr = periodogram


def cheatsheet() -> str:
    return "periodogram({}) -> Power spectral density via periodogram."
