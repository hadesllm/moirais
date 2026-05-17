"""
Livestock water quality

Category: WtrQual
"""

import numpy as np


def wqlvs(data=None, coords=None, n=50):
    """Livestock water quality

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


short = "wqlvs"
alias = "wqlvs"
quote = "The only true wisdom is in knowing you know nothing. -- Socrates"
wqlvs = wqlvs


def cheatsheet() -> str:
    return "wqlvs({}) -> Livestock water quality"
