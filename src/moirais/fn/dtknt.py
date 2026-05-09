# moirais.fn — function file (hadesllm/moirais)
"""
Kent distribution (spherical)

Category: DistTheor
"""

import numpy as np


def dtknt(x=None, n=100, params=None):
    """Kent distribution (spherical)

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


short = "dtknt"
alias = "dtknt"
quote = "Set your heart ablaze! -- Rengoku"
dtknt = dtknt


def cheatsheet() -> str:
    return "dtknt({}) -> Kent distribution (spherical)"
