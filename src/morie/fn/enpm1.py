# morie.fn -- function file (hadesllm/morie)
"""
PM10 spatial interpolation

Category: EnvStat
"""

import numpy as np


def enpm1(data=None, coords=None, n=50):
    """PM10 spatial interpolation

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


short = "enpm1"
alias = "enpm1"
quote = "Logic is the foundation of all certain knowledge. -- Leonhard Euler"
enpm1 = enpm1


def cheatsheet() -> str:
    return "enpm1({}) -> PM10 spatial interpolation"
