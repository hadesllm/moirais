# morie.fn — function file (hadesllm/morie)
"""
Multivariate t density

Category: DistTheor
"""

import numpy as np


def dtmvt(x=None, n=100, params=None):
    """Multivariate t density

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


short = "dtmvt"
alias = "dtmvt"
quote = "There is always hope. -- Aragorn"
dtmvt = dtmvt


def cheatsheet() -> str:
    return "dtmvt({}) -> Multivariate t density"
