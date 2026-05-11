# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""
PM fine spatial indoor

Category: AirBio
"""

import numpy as np


def abpmf(data=None, coords=None, n=50):
    """PM fine spatial indoor

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


short = "abpmf"
alias = "abpmf"
quote = "Make it so. -- Picard"
abpmf = abpmf


def cheatsheet() -> str:
    return "abpmf({}) -> PM fine spatial indoor"
