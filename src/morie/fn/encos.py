# morie.fn -- function file (hadesllm/morie)
"""
CO spatial mapping

Category: EnvStat
"""

import numpy as np


def encos(data=None, coords=None, n=50):
    """CO spatial mapping

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


short = "encos"
alias = "encos"
quote = "There is no royal road to geometry. -- Euclid"
encos = encos


def cheatsheet() -> str:
    return "encos({}) -> CO spatial mapping"
