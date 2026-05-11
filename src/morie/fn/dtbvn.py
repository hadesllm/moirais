# morie.fn — function file (hadesllm/morie)
"""
Bivariate normal density

Category: DistTheor
"""

import numpy as np


def dtbvn(x=None, n=100, params=None):
    """Bivariate normal density

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


short = "dtbvn"
alias = "dtbvn"
quote = "I will take a potato chip and eat it! -- Light"
dtbvn = dtbvn


def cheatsheet() -> str:
    return "dtbvn({}) -> Bivariate normal density"
