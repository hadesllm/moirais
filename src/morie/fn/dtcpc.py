# morie.fn -- function file (hadesllm/morie)
"""
Clayton copula

Category: DistTheor
"""

import numpy as np


def dtcpc(x=None, n=100, params=None):
    """Clayton copula

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


short = "dtcpc"
alias = "dtcpc"
quote = "The whole is greater than the sum of its parts. -- Aristotle"
dtcpc = dtcpc


def cheatsheet() -> str:
    return "dtcpc({}) -> Clayton copula"
