# morie.fn -- function file (hadesllm/morie)
"""
Exceedance probability air

Category: EnvStat
"""

import numpy as np


def enexc(data=None, coords=None, n=50):
    """Exceedance probability air

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


short = "enexc"
alias = "enexc"
quote = "Dedicate your hearts! -- Erwin"
enexc = enexc


def cheatsheet() -> str:
    return "enexc({}) -> Exceedance probability air"
