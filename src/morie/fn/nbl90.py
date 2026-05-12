# morie.fn -- function file (hadesllm/morie)
"""
L90 percentile level

Category: NoisBrd
"""

import numpy as np


def nbl90(data=None, coords=None, n=50):
    """L90 percentile level

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


short = "nbl90"
alias = "nbl90"
quote = "Believe it! -- Naruto"
nbl90 = nbl90


def cheatsheet() -> str:
    return "nbl90({}) -> L90 percentile level"
