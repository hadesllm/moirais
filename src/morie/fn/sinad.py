"""SINAD (signal to noise and distortion)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The greatest teacher, failure is."


def sinad_compute(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    r"""Compute SINAD (signal to noise and distortion ratio).

    .. math::

        \\text{SINAD} = 10 \\cdot \\log_{10}\\left(\\frac{P_{\\text{signal}}}{P_{\\text{noise+distortion}}}\\right)

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
    mag = np.abs(X)
    fund_idx = np.argmax(mag[1:]) + 1
    signal_power = mag[fund_idx] ** 2
    noise_dist_power = np.sum(mag**2) - signal_power
    if noise_dist_power <= 0:
        sinad_db = float("inf")
    else:
        sinad_db = 10.0 * np.log10(signal_power / noise_dist_power)
    return DescriptiveResult(
        name="sinad",
        value=float(sinad_db),
        extra={
            "sinad_db": float(sinad_db),
            "signal_power": float(signal_power),
            "noise_dist_power": float(noise_dist_power),
            "fund_bin": int(fund_idx),
            "fs": fs,
        },
    )


sinad = sinad_compute


def cheatsheet() -> str:
    return "sinad_compute({}) -> SINAD (signal to noise and distortion)."
