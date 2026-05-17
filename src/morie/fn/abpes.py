# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Pesticide spatial air

Category: AirBio
"""

import numpy as np


def abpes(data=None, coords=None, n=50):
    """Pesticide spatial air

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "abpes"
alias = "abpes"
quote = "The whole is greater than the sum of its parts. -- Aristotle"
abpes = abpes


def cheatsheet() -> str:
    return "abpes({}) -> Pesticide spatial air"
