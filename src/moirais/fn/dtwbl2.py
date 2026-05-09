# moirais.fn — function file (hadesllm/moirais)
"""
Weibull 3-parameter

Category: DistTheor
"""

import numpy as np


def dtwbl2(x=None, n=100, params=None):
    """Weibull 3-parameter

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


short = "dtwbl2"
alias = "dtwbl2"
quote = "A lesson without pain is meaningless. -- Edward"
dtwbl2 = dtwbl2


def cheatsheet() -> str:
    return "dtwbl2({}) -> Weibull 3-parameter"
