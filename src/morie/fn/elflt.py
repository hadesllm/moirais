# morie.fn -- function file (hadesllm/morie)
"""Elliptic (Cauer) filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "So this is how liberty dies... with thunderous applause."


def elliptic_filter(x, cutoff, fs, order: int = 4, rp: float = 1.0, rs: float = 40.0) -> SignalResult:
    r"""Apply an elliptic (Cauer) IIR filter to signal *x*.

    .. math::

        |H(j\\omega)|^2 = \\frac{1}{1 + \\varepsilon^2 R_N^2(\\xi, \\omega/\\omega_c)}

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
    rs : float
        Minimum stopband attenuation (dB). Default 40.0.

    Returns
    -------
    SignalResult
    """
    from scipy.signal import ellip, sosfiltfilt

    x = np.asarray(x, dtype=float)
    Wn = float(cutoff) / (fs / 2.0)
    sos = ellip(order, rp, rs, Wn, btype="low", output="sos")
    y = sosfiltfilt(sos, x)
    return SignalResult(
        name="elliptic_filter",
        filtered=y,
        fs=fs,
        n_samples=len(x),
        extra={"order": order, "cutoff": cutoff, "rp": rp, "rs": rs},
    )


elflt = elliptic_filter


def cheatsheet() -> str:
    return "elliptic_filter({}) -> Elliptic (Cauer) filter."
