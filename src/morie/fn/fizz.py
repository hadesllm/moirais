# morie.fn -- function file (hadesllm/morie)
"""FizzBuzz proportions."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def fizzbuzz_stats(n: int) -> DescriptiveResult:
    """Proportions of fizz, buzz, fizzbuzz, and other in 1..n.

    Parameters
    ----------
    n : int

    Returns
    -------
    DescriptiveResult
    """
    if n < 1:
        raise ValueError("n must be >= 1.")

    fb = np.sum(np.arange(1, n + 1) % 15 == 0)
    f = np.sum(np.arange(1, n + 1) % 3 == 0) - fb
    b = np.sum(np.arange(1, n + 1) % 5 == 0) - fb
    other = n - f - b - fb

    return DescriptiveResult(
        name="fizzbuzz",
        value=float(fb / n),
        extra={
            "n": n,
            "fizz_count": int(f),
            "buzz_count": int(b),
            "fizzbuzz_count": int(fb),
            "other_count": int(other),
            "fizz_pct": float(f / n * 100),
            "buzz_pct": float(b / n * 100),
            "fizzbuzz_pct": float(fb / n * 100),
        },
    )


fizz = fizzbuzz_stats


def cheatsheet() -> str:
    return "fizzbuzz_stats({}) -> FizzBuzz proportions."
