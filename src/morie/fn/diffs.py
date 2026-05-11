# morie.fn — function file (hadesllm/morie)
"""Numerical differentiation of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Difficult to see. Always in motion is the future."


def differentiate_signal(x, fs: float = 1.0, order: int = 1) -> SignalResult:
    """Compute the numerical derivative of signal *x*.

    .. math::

        x'(n) \\approx \\frac{x(n) - x(n-1)}{1/f_s}

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency (Hz). Default 1.0.
    order : int
        Derivative order. Default 1.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    y = x.copy()
    dt = 1.0 / fs
    for _ in range(order):
        y = np.gradient(y, dt)
    return SignalResult(
        name="differentiate_signal",
        filtered=y,
        fs=fs,
        n_samples=len(y),
        extra={"order": order},
    )


diffs = differentiate_signal


def cheatsheet() -> str:
    return "differentiate_signal({}) -> Numerical differentiation of a signal."
