"""
Isolation by environment

Category: WildlSp
"""

import numpy as np


def wlibe(abundance=None, coords=None, n=50):
    """Isolation by environment

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


short = "wlibe"
alias = "wlibe"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
wlibe = wlibe


def cheatsheet() -> str:
    return "wlibe({}) -> Isolation by environment"
