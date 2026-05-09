# moirais.fn — function file (hadesllm/moirais)
"""
Temporal trend air quality

Category: EnvStat
"""

import numpy as np


def entrd(data=None, coords=None, n=50):
    """Temporal trend air quality

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


short = "entrd"
alias = "entrd"
quote = "I am justice! -- Light"
entrd = entrd


def cheatsheet() -> str:
    return "entrd({}) -> Temporal trend air quality"
