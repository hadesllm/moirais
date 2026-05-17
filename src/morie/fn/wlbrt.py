"""
BRT species model

Category: WildlSp
"""

import numpy as np


def wlbrt(abundance=None, coords=None, n=50):
    """BRT species model

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


short = "wlbrt"
alias = "wlbrt"
quote = "Measure what is measurable, and make measurable what is not. -- Galileo Galilei"
wlbrt = wlbrt


def cheatsheet() -> str:
    return "wlbrt({}) -> BRT species model"
