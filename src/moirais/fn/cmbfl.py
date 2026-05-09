# moirais.fn — function file (hadesllm/moirais)
"""Comb filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Luminous beings are we, not this crude matter."


def comb_filter(x, delay: int, alpha: float = 0.5) -> SignalResult:
    """Apply a feedforward comb filter.

    .. math::

        y(n) = x(n) + \\alpha \\cdot x(n - D)

    Parameters
    ----------
    x : array-like
        Input signal.
    delay : int
        Delay in samples (D).
    alpha : float
        Gain coefficient. Default 0.5.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    delay = int(delay)
    y = np.copy(x)
    if delay > 0 and delay < len(x):
        y[delay:] += alpha * x[:-delay]
    return SignalResult(
        name="comb_filter",
        filtered=y,
        fs=0.0,
        n_samples=len(x),
        extra={"delay": delay, "alpha": alpha},
    )


cmbfl = comb_filter


def cheatsheet() -> str:
    return "comb_filter({}) -> Comb filter."
