# morie.fn -- function file (hadesllm/morie)
"""
Ordered logistic distribution

Category: DistTheor
"""

import numpy as np


def dtord(x=None, n=100, params=None):
    """Ordered logistic distribution

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


short = "dtord"
alias = "dtord"
quote = "There is no royal road to geometry. -- Euclid"
dtord = dtord


def cheatsheet() -> str:
    return "dtord({}) -> Ordered logistic distribution"
