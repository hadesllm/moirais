# morie.fn -- function file (hadesllm/morie)
"""
Drug market spatial

Category: CrimSp
"""

import numpy as np


def csdrg(incidents=None, population=None, coords=None, n=50):
    """Drug market spatial

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


short = "csdrg"
alias = "csdrg"
quote = "One does not simply walk. -- Boromir"
csdrg = csdrg


def cheatsheet() -> str:
    return "csdrg({}) -> Drug market spatial"
