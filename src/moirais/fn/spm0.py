"""Spectral moment."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The ability to speak does not make you intelligent."


def spectral_moment(x, fs: float = 1.0, k: int = 0, **kwargs) -> DescriptiveResult:
    """Compute the k-th spectral moment.

    .. math::

        m_k = \\int_0^{f_s/2} f^k \\cdot S(f) \\, df

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.
    k : int
        Order of the moment.

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
    mk = float(np.sum((freqs**k) * psd * df))
    return DescriptiveResult(
        name="spectral_moment",
        value=mk,
        extra={"moment_order": k, "mk": mk, "fs": fs},
    )


spm0 = spectral_moment


def cheatsheet() -> str:
    return "spectral_moment({}) -> Spectral moment."
