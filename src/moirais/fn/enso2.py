# moirais.fn — function file (hadesllm/moirais)
"""
SO2 concentration spatial

Category: EnvStat
"""

import numpy as np


def enso2(data=None, coords=None, n=50):
    """SO2 concentration spatial

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


short = "enso2"
alias = "enso2"
quote = "Believe it! -- Naruto"
enso2 = enso2


def cheatsheet() -> str:
    return "enso2({}) -> SO2 concentration spatial"
