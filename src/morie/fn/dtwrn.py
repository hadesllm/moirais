# morie.fn -- function file (hadesllm/morie)
"""
Wrapped normal distribution

Category: DistTheor
"""

import numpy as np


def dtwrn(x=None, n=100, params=None):
    """Wrapped normal distribution

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if x is None:
        x = np.random.default_rng(0).standard_normal(n)
    if params is None:
        params = {"loc": float(np.mean(x)), "scale": float(np.std(x))}
    stat = float(np.mean(x))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(x), "mean": float(np.mean(x)), "std": float(np.std(x)), "params": params},
    )


short = "dtwrn"
alias = "dtwrn"
quote = "I am the one who knocks. -- Walter White"
dtwrn = dtwrn


def cheatsheet() -> str:
    return "dtwrn({}) -> Wrapped normal distribution"
