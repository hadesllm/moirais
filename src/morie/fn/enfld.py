# morie.fn -- function file (hadesllm/morie)
"""
Flood risk spatial

Category: EnvStat
"""

import numpy as np


def enfld(data=None, coords=None, n=50):
    """Flood risk spatial

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


short = "enfld"
alias = "enfld"
quote = "Knowledge is power. -- Francis Bacon"
enfld = enfld


def cheatsheet() -> str:
    return "enfld({}) -> Flood risk spatial"
