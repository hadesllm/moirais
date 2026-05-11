"""
Least cost path wildlife

Category: WildlSp
"""

import numpy as np


def wllcp(abundance=None, coords=None, n=50):
    """Least cost path wildlife

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if abundance is None:
        abundance = np.random.default_rng(0).poisson(10, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(abundance))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(abundance), "total": int(np.sum(abundance)), "mean": float(np.mean(abundance))},
    )


short = "wllcp"
alias = "wllcp"
quote = "Not all those who wander are lost. -- Gandalf"
wllcp = wllcp


def cheatsheet() -> str:
    return "wllcp({}) -> Least cost path wildlife"
