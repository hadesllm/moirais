# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
PM coarse spatial indoor

Category: AirBio
"""

import numpy as np


def abpmc(data=None, coords=None, n=50):
    """PM coarse spatial indoor

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


short = "abpmc"
alias = "abpmc"
quote = "What is now proved was once only imagined. -- William Blake"
abpmc = abpmc


def cheatsheet() -> str:
    return "abpmc({}) -> PM coarse spatial indoor"
