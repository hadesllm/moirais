# morie.fn — function file (hadesllm/morie)
"""
CFD wind field spatial

Category: EnvStat
"""

import numpy as np


def encfd(data=None, coords=None, n=50):
    """CFD wind field spatial

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


short = "encfd"
alias = "encfd"
quote = "Winter is coming. -- Stark motto"
encfd = encfd


def cheatsheet() -> str:
    return "encfd({}) -> CFD wind field spatial"
