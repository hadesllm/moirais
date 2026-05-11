# morie.fn — function file (hadesllm/morie)
"""
Johnson SB distribution

Category: DistTheor
"""

import numpy as np


def dtjhm(x=None, n=100, params=None):
    """Johnson SB distribution

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


short = "dtjhm"
alias = "dtjhm"
quote = "One does not simply walk. -- Boromir"
dtjhm = dtjhm


def cheatsheet() -> str:
    return "dtjhm({}) -> Johnson SB distribution"
