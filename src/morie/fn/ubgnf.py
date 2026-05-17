"""
Gentrification risk spatial

Category: UrbanSp
"""

import numpy as np


def ubgnf(population=None, area=None, coords=None, n=50):
    """Gentrification risk spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if population is None:
        population = np.random.default_rng(0).poisson(5000, n)
    if area is None:
        area = np.random.default_rng(1).uniform(1, 100, n)
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    density = population / area
    stat = float(np.mean(density))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(population), "mean_density": float(np.mean(density)), "total_pop": int(np.sum(population))},
    )


short = "ubgnf"
alias = "ubgnf"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
ubgnf = ubgnf


def cheatsheet() -> str:
    return "ubgnf({}) -> Gentrification risk spatial"
