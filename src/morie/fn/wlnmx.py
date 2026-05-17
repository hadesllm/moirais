"""
N-mixture model spatial

Category: WildlSp
"""

import numpy as np


def wlnmx(abundance=None, coords=None, n=50):
    """N-mixture model spatial

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


short = "wlnmx"
alias = "wlnmx"
quote = "Mathematics is the queen of the sciences. -- Carl Friedrich Gauss"
wlnmx = wlnmx


def cheatsheet() -> str:
    return "wlnmx({}) -> N-mixture model spatial"
