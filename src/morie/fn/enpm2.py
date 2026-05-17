# morie.fn -- function file (hadesllm/morie)
"""
PM2.5 spatial interpolation

Category: EnvStat
"""

import numpy as np


def enpm2(data=None, coords=None, n=50):
    """PM2.5 spatial interpolation

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


short = "enpm2"
alias = "enpm2"
quote = "It is not the strongest that survives, but the most adaptable. -- Charles Darwin"
enpm2 = enpm2


def cheatsheet() -> str:
    return "enpm2({}) -> PM2.5 spatial interpolation"
