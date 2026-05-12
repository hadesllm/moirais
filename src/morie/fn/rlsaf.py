# morie.fn -- function file (hadesllm/morie)
"""RLS adaptive filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def rls_adaptive_filter(x, d, order: int = 16, lam: float = 0.99, delta: float = 100.0) -> SignalResult:
    """Apply an RLS (Recursive Least Squares) adaptive filter.

    Parameters
    ----------
    x : array-like
        Input (reference) signal.
    d : array-like
        Desired signal.
    order : int
        Filter order. Default 16.
    lam : float
        Forgetting factor (0 < lam <= 1). Default 0.99.
    delta : float
        Regularization parameter for initial correlation matrix. Default 100.0.

    Returns
    -------
    SignalResult
        *extra* contains ``output`` (y) and ``error`` (e) arrays.
    """
    from morie._filters import rls_filter as _rls

    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    y, e = _rls(x, d, order=order, lam=lam, delta=delta)
    return SignalResult(
        name="rls_adaptive_filter",
        filtered=y,
        fs=0.0,
        n_samples=len(x),
        extra={"output": y, "error": e},
    )


rlsaf = rls_adaptive_filter


def cheatsheet() -> str:
    return "rls_adaptive_filter({}) -> RLS adaptive filter."
