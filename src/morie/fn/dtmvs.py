# morie.fn -- function file (hadesllm/morie)
"""
Multivariate skew-normal

Category: DistTheor
"""

import numpy as np


def dtmvs(x=None, n=100, params=None):
    """Multivariate skew-normal

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


short = "dtmvs"
alias = "dtmvs"
quote = "Knowledge is power. -- Francis Bacon"
dtmvs = dtmvs


def cheatsheet() -> str:
    return "dtmvs({}) -> Multivariate skew-normal"
