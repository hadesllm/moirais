"""
Exchangeable sodium percentage

Category: SoilSp
"""

import numpy as np


def soesp(data=None, depth=None, coords=None, n=50):
    """Exchangeable sodium percentage

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


short = "soesp"
alias = "soesp"
quote = "Not all those who wander are lost. -- Gandalf"
soesp = soesp


def cheatsheet() -> str:
    return "soesp({}) -> Exchangeable sodium percentage"
