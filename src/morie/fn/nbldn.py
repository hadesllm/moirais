# morie.fn — function file (hadesllm/morie)
"""
Ldn day-night level

Category: NoisBrd
"""

import numpy as np


def nbldn(data=None, coords=None, n=50):
    """Ldn day-night level

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(30, 90, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean_db": float(np.mean(data)), "max_db": float(np.max(data))},
    )


short = "nbldn"
alias = "nbldn"
quote = "Winter is coming. -- Stark motto"
nbldn = nbldn


def cheatsheet() -> str:
    return "nbldn({}) -> Ldn day-night level"
