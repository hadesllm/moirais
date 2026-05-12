# morie.fn — function file (hadesllm/morie)
"""Chebyshev Type I filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "I find your lack of faith disturbing."


def chebyshev1_filter(x, cutoff, fs, order: int = 4, rp: float = 1.0) -> SignalResult:
    r"""Apply a Chebyshev Type I filter to signal *x*.

    .. math::

        |H(j\\omega)|^2 = \\frac{1}{1 + \\varepsilon^2 T_N^2(\\omega/\\omega_c)}

    Parameters
    ----------
    x : array-like
        Input signal.
    cutoff : float
        Cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order. Default 4.
    rp : float
        Maximum passband ripple (dB). Default 1.0.

    Returns
    -------
    SignalResult
    """
    from scipy.signal import cheby1, sosfiltfilt

    x = np.asarray(x, dtype=float)
    Wn = float(cutoff) / (fs / 2.0)
    sos = cheby1(order, rp, Wn, btype="low", output="sos")
    y = sosfiltfilt(sos, x)
    return SignalResult(
        name="chebyshev1_filter",
        filtered=y,
        fs=fs,
        n_samples=len(x),
        extra={"order": order, "cutoff": cutoff, "rp": rp},
    )


chbf1 = chebyshev1_filter


def cheatsheet() -> str:
    return "chebyshev1_filter({}) -> Chebyshev Type I filter."
