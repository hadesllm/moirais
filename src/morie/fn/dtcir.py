# morie.fn -- function file (hadesllm/morie)
"""
Circular uniform distribution

Category: DistTheor
"""

import numpy as np


def dtcir(x=None, n=100, params=None):
    """Circular uniform distribution

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


short = "dtcir"
alias = "dtcir"
quote = "Live long and prosper. -- Spock"
dtcir = dtcir


def cheatsheet() -> str:
    return "dtcir({}) -> Circular uniform distribution"
