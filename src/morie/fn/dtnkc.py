# morie.fn — function file (hadesllm/morie)
"""
Nakagami distribution

Category: DistTheor
"""

import numpy as np


def dtnkc(x=None, n=100, params=None):
    """Nakagami distribution

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


short = "dtnkc"
alias = "dtnkc"
quote = "Even the smallest person can change the future. -- Galadriel"
dtnkc = dtnkc


def cheatsheet() -> str:
    return "dtnkc({}) -> Nakagami distribution"
