# moirais.fn — function file (hadesllm/moirais)
"""Fibonacci golden ratio convergence."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def fibonacci_ratio(n: int) -> ESRes:
    """Compute F(n)/F(n-1) convergence to the golden ratio phi.

    Parameters
    ----------
    n : int (>= 3)

    Returns
    -------
    ESRes
    """
    if n < 3:
        raise ValueError("n must be >= 3.")

    a, b = 1, 1
    ratios = []
    for _ in range(n - 2):
        a, b = b, a + b
        ratios.append(b / a)

    phi = (1 + np.sqrt(5)) / 2
    final_ratio = ratios[-1]
    error = abs(final_ratio - phi)

    return ESRes(
        measure="fibonacci_ratio",
        estimate=float(final_ratio),
        n=n,
        extra={"golden_ratio": float(phi), "abs_error": float(error), "fib_n": int(b), "n_ratios": len(ratios)},
    )


fib = fibonacci_ratio


def cheatsheet() -> str:
    return "fibonacci_ratio({}) -> Fibonacci golden ratio convergence."
