# morie.fn — function file (hadesllm/morie)
"""Prony's method for pole-zero estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Rebellions are built on hope."


def prony_method_fn(x: np.ndarray, p: int = 4, q: int = 2) -> DescriptiveResult:
    """Estimate poles and zeros via Prony's method.

    Fits an ARMA(p,q) model by first estimating the AR denominator from
    the signal autocorrelation, then solving for the MA numerator.

    :param x: 1-D input signal (or impulse response).
    :param p: Number of poles (default 4).
    :param q: Number of zeros (default 2).
    :return: DescriptiveResult with AR/MA coefficients, poles, and zeros.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if p + q + 1 > n:
        raise ValueError("Signal too short for requested p and q")
    H = np.zeros((n - p, p))
    for i in range(n - p):
        for j in range(p):
            idx = p + i - j - 1
            if 0 <= idx < n:
                H[i, j] = x[idx]
    rhs = -x[p:n]
    try:
        a = np.linalg.lstsq(H[: len(rhs)], rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        a = np.zeros(p)
    b = np.zeros(q + 1)
    for i in range(min(q + 1, n)):
        b[i] = x[i]
        for j in range(min(i, p)):
            b[i] += a[j] * x[i - j - 1] if i - j - 1 >= 0 else 0
    ar_poly = np.concatenate(([1.0], a))
    ma_poly = b
    poles = np.roots(ar_poly) if len(ar_poly) > 1 else np.array([])
    zeros = np.roots(ma_poly) if len(ma_poly) > 1 else np.array([])
    return DescriptiveResult(
        name="prony_method",
        value=None,
        extra={
            "ar_coeffs": a,
            "ma_coeffs": b,
            "poles": poles,
            "zeros": zeros,
            "p": p,
            "q": q,
        },
    )


prdny = prony_method_fn


def cheatsheet() -> str:
    return "prony_method_fn({}) -> Prony's method for pole-zero estimation."
