# morie.fn — function file (hadesllm/morie)
"""LMS adaptive filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "We suffer more often in imagination than in reality. — Seneca"


def lms_filter(x, d, mu: float = 0.01, order: int = 16) -> SignalResult:
    """Least Mean Squares adaptive filter.

    .. math::

        w(n+1) = w(n) + 2\\mu \\, e(n) \\, x(n)

    Parameters
    ----------
    x : array-like
        Input (reference) signal.
    d : array-like
        Desired signal.
    mu : float
        Step size. Default 0.01.
    order : int
        Filter order (number of taps). Default 16.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    N = min(len(x), len(d))
    w = np.zeros(order)
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(order, N):
        xn = x[n - order : n][::-1]
        y[n] = np.dot(w, xn)
        e[n] = d[n] - y[n]
        w = w + 2 * mu * e[n] * xn
    return SignalResult(
        name="lms_filter",
        filtered=y,
        fs=0.0,
        n_samples=N,
        extra={"weights": w, "error": e, "mu": mu, "order": order},
    )


lmsfl = lms_filter


def cheatsheet() -> str:
    return "lms_filter({}) -> LMS adaptive filter."
