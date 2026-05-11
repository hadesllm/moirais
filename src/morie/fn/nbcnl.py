# morie.fn — function file (hadesllm/morie)
"""
CNEL community level

Category: NoisBrd
"""

import numpy as np


def nbcnl(data=None, coords=None, n=50):
    """CNEL community level

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


short = "nbcnl"
alias = "nbcnl"
quote = "It's over 9000! -- Vegeta"
nbcnl = nbcnl


def cheatsheet() -> str:
    return "nbcnl({}) -> CNEL community level"
