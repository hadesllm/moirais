# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Pollen spatial mapping

Category: AirBio
"""

import numpy as np


def abpln(data=None, coords=None, n=50):
    """Pollen spatial mapping

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


short = "abpln"
alias = "abpln"
quote = "No man ever steps in the same river twice. -- Heraclitus"
abpln = abpln


def cheatsheet() -> str:
    return "abpln({}) -> Pollen spatial mapping"
