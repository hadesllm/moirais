"""
BOD water spatial

Category: WtrQual
"""

import numpy as np


def wqbod(data=None, coords=None, n=50):
    """BOD water spatial

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


short = "wqbod"
alias = "wqbod"
quote = "The whole is greater than the sum of its parts. -- Aristotle"
wqbod = wqbod


def cheatsheet() -> str:
    return "wqbod({}) -> BOD water spatial"
