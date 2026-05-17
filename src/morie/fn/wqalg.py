"""
Algae bloom spatial

Category: WtrQual
"""

import numpy as np


def wqalg(data=None, coords=None, n=50):
    """Algae bloom spatial

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


short = "wqalg"
alias = "wqalg"
quote = "An investment in knowledge pays the best interest. -- Benjamin Franklin"
wqalg = wqalg


def cheatsheet() -> str:
    return "wqalg({}) -> Algae bloom spatial"
