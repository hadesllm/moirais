"""
GLM species model

Category: WildlSp
"""

import numpy as np


def wlglm(abundance=None, coords=None, n=50):
    """GLM species model

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


short = "wlglm"
alias = "wlglm"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
wlglm = wlglm


def cheatsheet() -> str:
    return "wlglm({}) -> GLM species model"
