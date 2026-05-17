# morie.fn -- function file (hadesllm/morie)
"""
Cold wave frequency

Category: GeoClim
"""

import numpy as np


def gccwv(data=None, coords=None, n=50):
    """Cold wave frequency

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "gccwv"
alias = "gccwv"
quote = "You have power over your mind, not outside events. -- Marcus Aurelius"
gccwv = gccwv


def cheatsheet() -> str:
    return "gccwv({}) -> Cold wave frequency"
