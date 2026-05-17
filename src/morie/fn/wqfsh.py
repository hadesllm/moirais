"""
Fishery water quality

Category: WtrQual
"""

import numpy as np


def wqfsh(data=None, coords=None, n=50):
    """Fishery water quality

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


short = "wqfsh"
alias = "wqfsh"
quote = "If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton"
wqfsh = wqfsh


def cheatsheet() -> str:
    return "wqfsh({}) -> Fishery water quality"
