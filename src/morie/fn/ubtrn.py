"""
Transit accessibility

Category: UrbanSp
"""

import numpy as np


def ubtrn(population=None, area=None, coords=None, n=50):
    """Transit accessibility

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


short = "ubtrn"
alias = "ubtrn"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
ubtrn = ubtrn


def cheatsheet() -> str:
    return "ubtrn({}) -> Transit accessibility"
