# morie.fn — function file (hadesllm/morie)
"""
Drought index spatial

Category: EnvStat
"""

import numpy as np


def endrt(data=None, coords=None, n=50):
    """Drought index spatial

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


short = "endrt"
alias = "endrt"
quote = "Engage. -- Picard"
endrt = endrt


def cheatsheet() -> str:
    return "endrt({}) -> Drought index spatial"
