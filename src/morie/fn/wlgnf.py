"""
Gene flow landscape

Category: WildlSp
"""

import numpy as np


def wlgnf(abundance=None, coords=None, n=50):
    """Gene flow landscape

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


short = "wlgnf"
alias = "wlgnf"
quote = "I'm gonna be King of the Pirates! -- Luffy"
wlgnf = wlgnf


def cheatsheet() -> str:
    return "wlgnf({}) -> Gene flow landscape"
