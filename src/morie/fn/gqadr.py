# morie.fn — function file (hadesllm/morie)
"""Gauss-Legendre quadrature."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you, it is ahead."


def gauss_quadrature(f, a, b, n=5, **kwargs) -> DescriptiveResult:
    """Integrate function *f* from *a* to *b* using Gauss-Legendre quadrature.

    Parameters
    ----------
    f : callable
        Function to integrate.
    a : float
        Lower bound.
    b : float
        Upper bound.
    n : int
        Number of quadrature points (default 5).

    Returns
    -------
    DescriptiveResult
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    mid = (b + a) / 2.0
    half = (b - a) / 2.0
    mapped = mid + half * nodes
    integral = float(half * np.dot(weights, f(mapped)))
    return DescriptiveResult(
        name="gauss_quadrature",
        value=integral,
        extra={
            "integral": integral,
            "a": a,
            "b": b,
            "n_points": n,
        },
    )


gqadr = gauss_quadrature


def cheatsheet() -> str:
    return "gauss_quadrature({}) -> Gauss-Legendre quadrature."
