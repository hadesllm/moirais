"""
Light pollution spatial

Category: UrbanSp
"""

import numpy as np


def ublit(population=None, area=None, coords=None, n=50):
    """Light pollution spatial

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


short = "ublit"
alias = "ublit"
quote = "Walk without rhythm. -- Fremen proverb"
ublit = ublit


def cheatsheet() -> str:
    return "ublit({}) -> Light pollution spatial"
