# morie.fn — function file (hadesllm/morie)
"""
Tornado risk spatial

Category: EnvStat
"""

import numpy as np


def entrd2(data=None, coords=None, n=50):
    """Tornado risk spatial

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


short = "entrd2"
alias = "entrd2"
quote = "You should enjoy the detours. -- Ging"
entrd2 = entrd2


def cheatsheet() -> str:
    return "entrd2({}) -> Tornado risk spatial"
