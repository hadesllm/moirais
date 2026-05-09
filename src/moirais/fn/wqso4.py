"""
Sulfate water spatial

Category: WtrQual
"""

import numpy as np


def wqso4(data=None, coords=None, n=50):
    """Sulfate water spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(0, 14, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "wqso4"
alias = "wqso4"
quote = "Dedicate your hearts! -- Erwin"
wqso4 = wqso4


def cheatsheet() -> str:
    return "wqso4({}) -> Sulfate water spatial"
