# morie.fn — function file (hadesllm/morie)
"""Normalized LMS adaptive filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "You were my brother, Anakin. I loved you."


def nlms_filter(x, d, mu: float = 0.5, order: int = 16) -> SignalResult:
    r"""Normalized Least Mean Squares adaptive filter.

    .. math::

        w(n+1) = w(n) + \\frac{\\mu}{\\|x(n)\\|^2 + \\delta} \\, e(n) \\, x(n)

    Parameters
    ----------
    x : array-like
        Input (reference) signal.
    d : array-like
        Desired signal.
    mu : float
        Step size (0 < mu < 2). Default 0.5.
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
    y = np.zeros(N)
    e = np.zeros(N)
    delta = 1e-8
    for n in range(order, N):
        xn = x[n - order : n][::-1]
        y[n] = np.dot(w, xn)
        e[n] = d[n] - y[n]
        norm = np.dot(xn, xn) + delta
        w = w + (mu / norm) * e[n] * xn
    return SignalResult(
        name="nlms_filter",
        filtered=y,
        fs=0.0,
        n_samples=N,
        extra={"weights": w, "error": e, "mu": mu, "order": order},
    )


nlmsf = nlms_filter


def cheatsheet() -> str:
    return "nlms_filter({}) -> Normalized LMS adaptive filter."
