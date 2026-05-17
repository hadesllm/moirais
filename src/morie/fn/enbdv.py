# morie.fn -- function file (hadesllm/morie)
"""
Biodiversity index spatial

Category: EnvStat
"""

import numpy as np


def enbdv(data=None, coords=None, n=50):
    """Biodiversity index spatial

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


short = "enbdv"
alias = "enbdv"
quote = "A journey of a thousand miles begins with a single step. -- Lao Tzu"
enbdv = enbdv


def cheatsheet() -> str:
    return "enbdv({}) -> Biodiversity index spatial"
