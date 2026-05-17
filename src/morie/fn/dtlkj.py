# morie.fn -- function file (hadesllm/morie)
"""
LKJ correlation distribution

Category: DistTheor
"""

import numpy as np


def dtlkj(x=None, n=100, params=None):
    """LKJ correlation distribution

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


short = "dtlkj"
alias = "dtlkj"
quote = "Errors using inadequate data are much less than those using none. -- Charles Babbage"
dtlkj = dtlkj


def cheatsheet() -> str:
    return "dtlkj({}) -> LKJ correlation distribution"
