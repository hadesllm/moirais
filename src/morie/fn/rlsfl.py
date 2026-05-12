# morie.fn — function file (hadesllm/morie)
"""RLS adaptive filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Power! Unlimited power!"


def rls_filter(x, d, lam: float = 0.99, order: int = 16) -> SignalResult:
    r"""Recursive Least Squares adaptive filter.

    .. math::

        w(n) = w(n-1) + k(n) \\, [d(n) - w^T(n-1) x(n)]

    Parameters
    ----------
    x : array-like
        Input (reference) signal.
    d : array-like
        Desired signal.
    lam : float
        Forgetting factor (0 < lam <= 1). Default 0.99.
    order : int
        Filter order. Default 16.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    N = min(len(x), len(d))
    w = np.zeros(order)
    P = np.eye(order) * 1000.0
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(order, N):
        xn = x[n - order : n][::-1]
        y[n] = np.dot(w, xn)
        e[n] = d[n] - y[n]
        Px = P @ xn
        denom = lam + np.dot(xn, Px)
        k = Px / denom
        w = w + k * e[n]
        P = (P - np.outer(k, Px)) / lam
    return SignalResult(
        name="rls_filter",
        filtered=y,
        fs=0.0,
        n_samples=N,
        extra={"weights": w, "error": e, "lambda": lam, "order": order},
    )


rlsfl = rls_filter


def cheatsheet() -> str:
    return "rls_filter({}) -> RLS adaptive filter."
