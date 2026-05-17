"""
Isolation by resistance

Category: WildlSp
"""

import numpy as np


def wlibr(abundance=None, coords=None, n=50):
    """Isolation by resistance

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


short = "wlibr"
alias = "wlibr"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
wlibr = wlibr


def cheatsheet() -> str:
    return "wlibr({}) -> Isolation by resistance"
