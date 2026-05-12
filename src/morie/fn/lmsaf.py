# morie.fn -- function file (hadesllm/morie)
"""LMS adaptive filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def lms_adaptive_filter(x, d, order: int = 16, mu: float = 0.01) -> SignalResult:
    """Apply an LMS (Least Mean Squares) adaptive filter.

    Parameters
    ----------
    x : array-like
        Input (reference) signal.
    d : array-like
        Desired signal.
    order : int
        Filter order. Default 16.
    mu : float
        Step size / learning rate. Default 0.01.

    Returns
    -------
    SignalResult
        *extra* contains ``output`` (y) and ``error`` (e) arrays.
    """
    from morie._filters import lms_filter as _lms

    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    y, e = _lms(x, d, order=order, mu=mu)
    return SignalResult(
        name="lms_adaptive_filter",
        filtered=y,
        fs=0.0,
        n_samples=len(x),
        extra={"output": y, "error": e},
    )


lmsaf = lms_adaptive_filter


def cheatsheet() -> str:
    return "lms_adaptive_filter({}) -> LMS adaptive filter."
