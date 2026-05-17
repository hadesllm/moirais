# morie.fn -- function file (hadesllm/morie)
"""
Bird diversity noise

Category: NoisBrd
"""

import numpy as np


def nbbrd(data=None, coords=None, n=50):
    """Bird diversity noise

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


short = "nbbrd"
alias = "nbbrd"
quote = "Knowledge is power. -- Francis Bacon"
nbbrd = nbbrd


def cheatsheet() -> str:
    return "nbbrd({}) -> Bird diversity noise"
