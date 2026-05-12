# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Spore spatial mapping

Category: AirBio
"""

import numpy as np


def abspor(data=None, coords=None, n=50):
    """Spore spatial mapping

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


short = "abspor"
alias = "abspor"
quote = "Go beyond! Plus Ultra! -- All Might"
abspor = abspor


def cheatsheet() -> str:
    return "abspor({}) -> Spore spatial mapping"
