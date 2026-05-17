"""
GAM species model

Category: WildlSp
"""

import numpy as np


def wlgam(abundance=None, coords=None, n=50):
    """GAM species model

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


short = "wlgam"
alias = "wlgam"
quote = "The Analytical Engine weaves algebraic patterns. -- Ada Lovelace"
wlgam = wlgam


def cheatsheet() -> str:
    return "wlgam({}) -> GAM species model"
