"""
Total phosphorus soil

Category: SoilSp
"""

import numpy as np


def sophs(data=None, depth=None, coords=None, n=50):
    """Total phosphorus soil

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(0, 1, n)
    if depth is None:
        depth = np.random.default_rng(1).uniform(0, 2, n)
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "mean_depth": float(np.mean(depth))},
    )


short = "sophs"
alias = "sophs"
quote = "An investment in knowledge pays the best interest. -- Benjamin Franklin"
sophs = sophs


def cheatsheet() -> str:
    return "sophs({}) -> Total phosphorus soil"
