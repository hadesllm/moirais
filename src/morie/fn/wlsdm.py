"""
Species distribution model

Category: WildlSp
"""

import numpy as np


def wlsdm(abundance=None, coords=None, n=50):
    """Species distribution model

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


short = "wlsdm"
alias = "wlsdm"
quote = "No man ever steps in the same river twice. -- Heraclitus"
wlsdm = wlsdm


def cheatsheet() -> str:
    return "wlsdm({}) -> Species distribution model"
