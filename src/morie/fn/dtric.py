# morie.fn -- function file (hadesllm/morie)
"""
Rice distribution

Category: DistTheor
"""

import numpy as np


def dtric(x=None, n=100, params=None):
    """Rice distribution

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


short = "dtric"
alias = "dtric"
quote = "I alone level up. -- Sung Jin-Woo"
dtric = dtric


def cheatsheet() -> str:
    return "dtric({}) -> Rice distribution"
