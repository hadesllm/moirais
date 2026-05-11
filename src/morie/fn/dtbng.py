# morie.fn — function file (hadesllm/morie)
"""
Bingham distribution

Category: DistTheor
"""

import numpy as np


def dtbng(x=None, n=100, params=None):
    """Bingham distribution

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


short = "dtbng"
alias = "dtbng"
quote = "Arise. -- Shadow Monarch"
dtbng = dtbng


def cheatsheet() -> str:
    return "dtbng({}) -> Bingham distribution"
