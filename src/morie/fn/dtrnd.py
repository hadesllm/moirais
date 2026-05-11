# morie.fn — function file (hadesllm/morie)
"""Polynomial detrending."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Who's the more foolish, the fool or the fool who follows him?"


def detrend_signal(x, order: int = 1) -> SignalResult:
    """Remove polynomial trend from signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.
    order : int
        Polynomial order to remove. Default 1 (linear).

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    n = np.arange(len(x), dtype=float)
    coeffs = np.polyfit(n, x, order)
    trend = np.polyval(coeffs, n)
    y = x - trend
    return SignalResult(
        name="detrend_signal",
        filtered=y,
        fs=0.0,
        n_samples=len(x),
        extra={"order": order, "trend_coeffs": coeffs},
    )


dtrnd = detrend_signal


def cheatsheet() -> str:
    return "detrend_signal({}) -> Polynomial detrending."
