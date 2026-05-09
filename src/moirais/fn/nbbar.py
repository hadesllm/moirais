# moirais.fn — function file (hadesllm/moirais)
"""
Barrier attenuation

Category: NoisBrd
"""

import numpy as np


def nbbar(data=None, coords=None, n=50):
    """Barrier attenuation

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


short = "nbbar"
alias = "nbbar"
quote = "I am justice! -- Light"
nbbar = nbbar


def cheatsheet() -> str:
    return "nbbar({}) -> Barrier attenuation"
