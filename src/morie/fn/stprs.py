"""Step response of a discrete-time system."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "The whole is greater than the sum of its parts. -- Aristotle"


def step_response(b, a, N: int = 100) -> SignalResult:
    """Compute the step response s(n) = cumsum(h(n)).

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
    step_input = np.ones(N)
    s = lfilter(b, a, step_input)
    return SignalResult(
        name="step_response",
        filtered=s,
        fs=0.0,
        n_samples=N,
        extra={"b": b, "a": a},
    )


stprs = step_response


def cheatsheet() -> str:
    return "step_response({}) -> Step response of a discrete-time system."
