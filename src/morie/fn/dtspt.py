# morie.fn -- function file (hadesllm/morie)
"""
Spatial extreme value

Category: DistTheor
"""

import numpy as np


def dtspt(x=None, n=100, params=None):
    """Spatial extreme value

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


short = "dtspt"
alias = "dtspt"
quote = "Mathematics is the queen of the sciences. -- Carl Friedrich Gauss"
dtspt = dtspt


def cheatsheet() -> str:
    return "dtspt({}) -> Spatial extreme value"
