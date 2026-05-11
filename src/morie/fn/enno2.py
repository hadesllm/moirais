# morie.fn — function file (hadesllm/morie)
"""
NO2 concentration spatial

Category: EnvStat
"""

import numpy as np


def enno2(data=None, coords=None, n=50):
    """NO2 concentration spatial

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


short = "enno2"
alias = "enno2"
quote = "Make it so. -- Picard"
enno2 = enno2


def cheatsheet() -> str:
    return "enno2({}) -> NO2 concentration spatial"
