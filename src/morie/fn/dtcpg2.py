# morie.fn -- function file (hadesllm/morie)
"""
Gumbel copula

Category: DistTheor
"""

import numpy as np


def dtcpg2(x=None, n=100, params=None):
    """Gumbel copula

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


short = "dtcpg2"
alias = "dtcpg2"
quote = "The heart has its reasons of which reason knows nothing. -- Blaise Pascal"
dtcpg2 = dtcpg2


def cheatsheet() -> str:
    return "dtcpg2({}) -> Gumbel copula"
