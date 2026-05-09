"""
Treatment plant performance

Category: WtrQual
"""

import numpy as np


def wqtrp(data=None, coords=None, n=50):
    """Treatment plant performance

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


short = "wqtrp"
alias = "wqtrp"
quote = "Dedicate your hearts! -- Erwin"
wqtrp = wqtrp


def cheatsheet() -> str:
    return "wqtrp({}) -> Treatment plant performance"
