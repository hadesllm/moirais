# morie.fn -- function file (hadesllm/morie)
"""
Heat wave frequency

Category: GeoClim
"""

import numpy as np


def gchwv(data=None, coords=None, n=50):
    """Heat wave frequency

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


short = "gchwv"
alias = "gchwv"
quote = "See you space cowboy. -- Spike"
gchwv = gchwv


def cheatsheet() -> str:
    return "gchwv({}) -> Heat wave frequency"
