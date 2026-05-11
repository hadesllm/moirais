"""
Total dissolved solids

Category: WtrQual
"""

import numpy as np


def wqtds(data=None, coords=None, n=50):
    """Total dissolved solids

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(0, 14, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "wqtds"
alias = "wqtds"
quote = "I'm gonna be King of the Pirates! -- Luffy"
wqtds = wqtds


def cheatsheet() -> str:
    return "wqtds({}) -> Total dissolved solids"
