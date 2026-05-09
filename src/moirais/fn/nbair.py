# moirais.fn — function file (hadesllm/moirais)
"""
Aircraft noise mapping

Category: NoisBrd
"""

import numpy as np


def nbair(data=None, coords=None, n=50):
    """Aircraft noise mapping

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(30, 90, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean_db": float(np.mean(data)), "max_db": float(np.max(data))},
    )


short = "nbair"
alias = "nbair"
quote = "Go beyond! Plus Ultra! -- All Might"
nbair = nbair


def cheatsheet() -> str:
    return "nbair({}) -> Aircraft noise mapping"
