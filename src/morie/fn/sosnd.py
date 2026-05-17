"""
Sand content spatial

Category: SoilSp
"""

import numpy as np


def sosnd(data=None, depth=None, coords=None, n=50):
    """Sand content spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(0, 1, n)
    if depth is None:
        depth = np.random.default_rng(1).uniform(0, 2, n)
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "mean_depth": float(np.mean(depth))},
    )


short = "sosnd"
alias = "sosnd"
quote = "What is now proved was once only imagined. -- William Blake"
sosnd = sosnd


def cheatsheet() -> str:
    return "sosnd({}) -> Sand content spatial"
