# morie.fn — function file (hadesllm/morie)
"""
L10 percentile level

Category: NoisBrd
"""

import numpy as np


def nbl10(data=None, coords=None, n=50):
    """L10 percentile level

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


short = "nbl10"
alias = "nbl10"
quote = "Not all those who wander are lost. -- Gandalf"
nbl10 = nbl10


def cheatsheet() -> str:
    return "nbl10({}) -> L10 percentile level"
