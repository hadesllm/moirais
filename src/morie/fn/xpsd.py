"""Cross power spectral density."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we, not this crude matter."


def cross_psd(x, y, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    r"""Compute the cross power spectral density of *x* and *y*.

    .. math::

        S_{xy}(f) = \\frac{1}{N \\cdot f_s} X^*(f) \\cdot Y(f)

    Parameters
    ----------
    x, y : array-like
        Input signals (same length).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = min(len(x), len(y))
    X = np.fft.rfft(x[:N])
    Y = np.fft.rfft(y[:N])
    cpsd = (np.conj(X) * Y) / (N * fs)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return DescriptiveResult(
        name="cross_psd",
        value=float(np.max(np.abs(cpsd))),
        extra={"cpsd": cpsd, "freqs": freqs, "fs": fs, "N": N},
    )


xpsd = cross_psd


def cheatsheet() -> str:
    return "cross_psd({}) -> Cross power spectral density."
