# morie.fn — function file (hadesllm/morie)
"""
Forest fragmentation spatial

Category: EnvStat
"""

import numpy as np


def enfrg(data=None, coords=None, n=50):
    """Forest fragmentation spatial

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


short = "enfrg"
alias = "enfrg"
quote = "No half measures. -- Mike"
enfrg = enfrg


def cheatsheet() -> str:
    return "enfrg({}) -> Forest fragmentation spatial"
