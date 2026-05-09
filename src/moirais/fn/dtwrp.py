# moirais.fn — function file (hadesllm/moirais)
"""
Wrapped Cauchy distribution

Category: DistTheor
"""

import numpy as np


def dtwrp(x=None, n=100, params=None):
    """Wrapped Cauchy distribution

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


short = "dtwrp"
alias = "dtwrp"
quote = "Winter is coming. -- Stark motto"
dtwrp = dtwrp


def cheatsheet() -> str:
    return "dtwrp({}) -> Wrapped Cauchy distribution"
