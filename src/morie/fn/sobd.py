"""
Bulk density soil

Category: SoilSp
"""

import numpy as np


def sobd(data=None, depth=None, coords=None, n=50):
    """Bulk density soil

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


short = "sobd"
alias = "sobd"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
sobd = sobd


def cheatsheet() -> str:
    return "sobd({}) -> Bulk density soil"
