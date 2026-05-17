# morie.fn -- function file (hadesllm/morie)
"""
Social disorganization spatial

Category: CrimSp
"""

import numpy as np


def cssfd(incidents=None, population=None, coords=None, n=50):
    """Social disorganization spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if incidents is None:
        incidents = np.random.default_rng(0).poisson(20, n)
    if population is None:
        population = np.random.default_rng(1).poisson(5000, n) + 100
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    rates = incidents / population * 1000
    stat = float(np.mean(rates))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(incidents), "total_incidents": int(np.sum(incidents)), "mean_rate": float(np.mean(rates))},
    )


short = "cssfd"
alias = "cssfd"
quote = "The heart has its reasons of which reason knows nothing. -- Blaise Pascal"
cssfd = cssfd


def cheatsheet() -> str:
    return "cssfd({}) -> Social disorganization spatial"
