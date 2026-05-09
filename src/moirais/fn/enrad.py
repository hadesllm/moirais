# moirais.fn — function file (hadesllm/moirais)
"""
Solar radiation spatial

Category: EnvStat
"""

import numpy as np


def enrad(data=None, coords=None, n=50):
    """Solar radiation spatial

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


short = "enrad"
alias = "enrad"
quote = "Breathe. -- Tanjiro"
enrad = enrad


def cheatsheet() -> str:
    return "enrad({}) -> Solar radiation spatial"
