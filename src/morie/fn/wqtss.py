"""
Total suspended solids water

Category: WtrQual
"""

import numpy as np


def wqtss(data=None, coords=None, n=50):
    """Total suspended solids water

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


short = "wqtss"
alias = "wqtss"
quote = "What is now proved was once only imagined. -- William Blake"
wqtss = wqtss


def cheatsheet() -> str:
    return "wqtss({}) -> Total suspended solids water"
