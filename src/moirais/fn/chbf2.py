# moirais.fn — function file (hadesllm/moirais)
"""Chebyshev Type II filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Never tell me the odds."


def chebyshev2_filter(x, cutoff, fs, order: int = 4, rs: float = 40.0) -> SignalResult:
    """Apply a Chebyshev Type II filter to signal *x*.

    .. math::

        |H(j\\omega)|^2 = \\frac{1}{1 + [\\varepsilon^2 T_N^2(\\omega_c/\\omega)]^{-1}}

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
    rs : float
        Minimum stopband attenuation (dB). Default 40.0.

    Returns
    -------
    SignalResult
    """
    from scipy.signal import cheby2, sosfiltfilt

    x = np.asarray(x, dtype=float)
    Wn = float(cutoff) / (fs / 2.0)
    sos = cheby2(order, rs, Wn, btype="low", output="sos")
    y = sosfiltfilt(sos, x)
    return SignalResult(
        name="chebyshev2_filter",
        filtered=y,
        fs=fs,
        n_samples=len(x),
        extra={"order": order, "cutoff": cutoff, "rs": rs},
    )


chbf2 = chebyshev2_filter


def cheatsheet() -> str:
    return "chebyshev2_filter({}) -> Chebyshev Type II filter."
