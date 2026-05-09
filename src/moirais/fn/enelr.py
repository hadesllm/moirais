# moirais.fn — function file (hadesllm/moirais)
"""
Eulerian dispersion

Category: EnvStat
"""

import numpy as np


def enelr(data=None, coords=None, n=50):
    """Eulerian dispersion

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


short = "enelr"
alias = "enelr"
quote = "Arise. -- Shadow Monarch"
enelr = enelr


def cheatsheet() -> str:
    return "enelr({}) -> Eulerian dispersion"
