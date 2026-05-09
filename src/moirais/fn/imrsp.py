# moirais.fn — function file (hadesllm/moirais)
"""Impulse response of a discrete-time system."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "You must unlearn what you have learned."


def impulse_response(b, a, N: int = 100) -> SignalResult:
    """Compute the impulse response h(n) from filter coefficients.

    Parameters
    ----------
    b : array-like
        Numerator coefficients.
    a : array-like
        Denominator coefficients.
    N : int
        Number of output samples. Default 100.

    Returns
    -------
    SignalResult
    """
    from scipy.signal import lfilter

    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    impulse = np.zeros(N)
    impulse[0] = 1.0
    h = lfilter(b, a, impulse)
    return SignalResult(
        name="impulse_response",
        filtered=h,
        fs=0.0,
        n_samples=N,
        extra={"b": b, "a": a},
    )


imrsp = impulse_response


def cheatsheet() -> str:
    return "impulse_response({}) -> Impulse response of a discrete-time system."
