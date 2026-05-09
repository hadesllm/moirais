# moirais.fn — function file (hadesllm/moirais)
"""
Call load distribution

Category: CrimSp
"""

import numpy as np


def cscld(incidents=None, population=None, coords=None, n=50):
    """Call load distribution

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if incidents is None:
        incidents = np.random.default_rng(0).poisson(20, n)
    if population is None:
        population = np.random.default_rng(1).poisson(5000, n) + 100
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    rates = incidents / population * 1000
    stat = float(np.mean(rates))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(incidents), "total_incidents": int(np.sum(incidents)), "mean_rate": float(np.mean(rates))},
    )


short = "cscld"
alias = "cscld"
quote = "Not all those who wander are lost. -- Gandalf"
cscld = cscld


def cheatsheet() -> str:
    return "cscld({}) -> Call load distribution"
