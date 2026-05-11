"""
Trend analysis species

Category: WildlSp
"""

import numpy as np


def wltrn(abundance=None, coords=None, n=50):
    """Trend analysis species

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


short = "wltrn"
alias = "wltrn"
quote = "See you space cowboy. -- Spike"
wltrn = wltrn


def cheatsheet() -> str:
    return "wltrn({}) -> Trend analysis species"
