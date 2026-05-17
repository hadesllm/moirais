"""
Capture-mark-recapture

Category: WildlSp
"""

import numpy as np


def wlcmr(abundance=None, coords=None, n=50):
    """Capture-mark-recapture

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


short = "wlcmr"
alias = "wlcmr"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
wlcmr = wlcmr


def cheatsheet() -> str:
    return "wlcmr({}) -> Capture-mark-recapture"
