# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Odor spatial mapping

Category: AirBio
"""

import numpy as np


def abodr(data=None, coords=None, n=50):
    """Odor spatial mapping

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


short = "abodr"
alias = "abodr"
quote = "I am the one who knocks. -- Walter White"
abodr = abodr


def cheatsheet() -> str:
    return "abodr({}) -> Odor spatial mapping"
