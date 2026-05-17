# morie.fn -- function file (hadesllm/morie)
"""
Wind rose spatial

Category: EnvStat
"""

import numpy as np


def enwnd(data=None, coords=None, n=50):
    """Wind rose spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "enwnd"
alias = "enwnd"
quote = "The Analytical Engine weaves algebraic patterns. -- Ada Lovelace"
enwnd = enwnd


def cheatsheet() -> str:
    return "enwnd({}) -> Wind rose spatial"
